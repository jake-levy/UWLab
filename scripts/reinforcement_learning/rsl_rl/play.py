# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_fps",
    type=int,
    default=None,
    help="Playback FPS for recorded videos. Defaults to the environment step rate if unset.",
)
parser.add_argument(
    "--video_size",
    nargs=2,
    type=int,
    metavar=("WIDTH", "HEIGHT"),
    default=None,
    help="Resolution for recorded videos as WIDTH HEIGHT. Defaults to the task viewer resolution if unset.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--autoreset",
    action="store_true",
    default=False,
    help="Automatically reset completed OmniReset play environments on success.",
)
parser.add_argument(
    "--zoom-out-vid",
    nargs=2,
    type=int,
    metavar=("START_FRAMES", "DURATION_FRAMES"),
    default=None,
    help="For recorded videos, hold the initial camera for START_FRAMES, then linearly zoom out for DURATION_FRAMES.",
)
parser.add_argument(
    "--zoom-out-start-eye-delta",
    nargs=3,
    type=float,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Offset applied to the zoom-out video start eye position. The end eye is recomputed to preserve view direction.",
)
parser.add_argument(
    "--zoom-out-start-spherical",
    nargs=3,
    type=float,
    metavar=("AZIMUTH_DEG", "ALTITUDE_DEG", "DISTANCE"),
    default=None,
    help="Alternative way to set the zoom-out video start eye from azimuth, altitude, and distance relative to lookat.",
)
parser.add_argument(
    "--zoom-out-start-origin-delta",
    nargs=3,
    type=float,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Offset applied to the spherical origin used by --zoom-out-start-spherical.",
)
parser.add_argument(
    "--zoom-out-distance-delta",
    type=float,
    default=0.0,
    help="Additional distance added to the computed final zoom-out camera distance.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as omnireset_mdp
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _compute_zoom_out_camera_poses(
    env, viewer_cfg, start_eye_delta=None, start_spherical=None, start_origin_delta=None, distance_delta=0.0
):
    """Compute the start and end camera poses for zoom-out video capture."""
    if viewer_cfg.origin_type != "world":
        raise ValueError("--zoom-out-vid currently only supports tasks with viewer.origin_type='world'.")

    start_lookat = np.asarray(viewer_cfg.lookat, dtype=float)
    if start_origin_delta is not None:
        start_lookat = start_lookat + np.asarray(start_origin_delta, dtype=float)
    if start_spherical is not None:
        azimuth_deg, altitude_deg, distance = start_spherical
        if distance <= 0.0:
            raise ValueError("DISTANCE for --zoom-out-start-spherical must be > 0.")
        azimuth = np.deg2rad(azimuth_deg)
        altitude = np.deg2rad(altitude_deg)
        start_eye = start_lookat + np.array(
            [
                distance * np.cos(altitude) * np.cos(azimuth),
                distance * np.cos(altitude) * np.sin(azimuth),
                distance * np.sin(altitude),
            ],
            dtype=float,
        )
    else:
        start_eye = np.asarray(viewer_cfg.eye, dtype=float)
    if start_eye_delta is not None:
        start_eye = start_eye + np.asarray(start_eye_delta, dtype=float)

    env_origins = env.unwrapped.scene.env_origins.detach().cpu().numpy()
    env_xy_min = env_origins[:, :2].min(axis=0)
    env_xy_max = env_origins[:, :2].max(axis=0)
    env_xy_center = 0.5 * (env_xy_min + env_xy_max)
    all_envs_center = np.array([env_xy_center[0], env_xy_center[1], start_lookat[2]], dtype=float)

    base_view_vec = start_eye - start_lookat
    base_distance = float(np.linalg.norm(base_view_vec))
    if base_distance <= 1e-6:
        raise ValueError("Viewer eye and lookat cannot be identical when using --zoom-out-vid.")

    span_xy = env_xy_max - env_xy_min
    max_span = float(max(span_xy[0], span_xy[1]))
    zoom_scale = max(1.0, 1.0 + max_span / 1.5)

    end_distance = max(base_distance, base_distance * zoom_scale + distance_delta)
    end_eye = all_envs_center + (base_view_vec / base_distance) * end_distance
    end_eye[2] = max(end_eye[2], start_eye[2] + 0.25 * max_span)
    end_lookat = all_envs_center
    return start_eye, start_lookat, end_eye, end_lookat


def _set_video_camera_pose(env, eye, lookat):
    """Set the viewport camera pose used for recorded rgb_array frames."""
    env.unwrapped.sim.set_camera_view(eye=eye.tolist(), target=lookat.tolist())


def _interpolate_camera_pose(frame_idx, start_frames, duration_frames, start_eye, start_lookat, end_eye, end_lookat):
    """Return the camera pose to use for a given recorded frame."""
    if frame_idx < start_frames:
        return start_eye, start_lookat
    if duration_frames <= 0 or frame_idx >= start_frames + duration_frames:
        return end_eye, end_lookat

    alpha = (frame_idx - start_frames + 1) / duration_frames
    alpha = min(max(alpha, 0.0), 1.0)
    eye = (1.0 - alpha) * start_eye + alpha * end_eye
    lookat = (1.0 - alpha) * start_lookat + alpha * end_lookat
    return eye, lookat


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.autoreset:
        if "OmniReset" not in task_name:
            raise ValueError("--autoreset is currently only supported for OmniReset tasks.")
        env_cfg.terminations.success = DoneTerm(
            func=omnireset_mdp.consecutive_success_state_with_min_length,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )
    if args_cli.zoom_out_vid is not None:
        if not args_cli.video:
            raise ValueError("--zoom-out-vid requires --video.")
        if args_cli.zoom_out_vid[0] < 0:
            raise ValueError("START_FRAMES for --zoom-out-vid must be >= 0.")
        if args_cli.zoom_out_vid[1] <= 0:
            raise ValueError("DURATION_FRAMES for --zoom-out-vid must be > 0.")
        if args_cli.zoom_out_vid[0] + args_cli.zoom_out_vid[1] > args_cli.video_length:
            raise ValueError("START_FRAMES + DURATION_FRAMES for --zoom-out-vid must be <= --video_length.")
    if args_cli.zoom_out_start_eye_delta is not None and args_cli.zoom_out_vid is None:
        raise ValueError("--zoom-out-start-eye-delta requires --zoom-out-vid.")
    if args_cli.zoom_out_start_spherical is not None and args_cli.zoom_out_vid is None:
        raise ValueError("--zoom-out-start-spherical requires --zoom-out-vid.")
    if args_cli.zoom_out_start_origin_delta is not None and args_cli.zoom_out_vid is None:
        raise ValueError("--zoom-out-start-origin-delta requires --zoom-out-vid.")
    if args_cli.zoom_out_start_eye_delta is not None and args_cli.zoom_out_start_spherical is not None:
        raise ValueError("--zoom-out-start-eye-delta and --zoom-out-start-spherical are mutually exclusive.")
    if args_cli.zoom_out_start_origin_delta is not None and args_cli.zoom_out_start_spherical is None:
        raise ValueError("--zoom-out-start-origin-delta requires --zoom-out-start-spherical.")
    if args_cli.zoom_out_distance_delta != 0.0 and args_cli.zoom_out_vid is None:
        raise ValueError("--zoom-out-distance-delta requires --zoom-out-vid.")
    if args_cli.video_fps is not None and args_cli.video_fps <= 0:
        raise ValueError("--video_fps must be > 0.")
    if args_cli.video_size is not None:
        if not args_cli.video:
            raise ValueError("--video_size requires --video.")
        if args_cli.video_size[0] <= 0 or args_cli.video_size[1] <= 0:
            raise ValueError("WIDTH and HEIGHT for --video_size must be > 0.")

        env_cfg.viewer.resolution = (args_cli.video_size[0], args_cli.video_size[1])

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "fps": args_cli.video_fps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    zoom_camera_poses = None
    if args_cli.zoom_out_vid is not None:
        zoom_camera_poses = _compute_zoom_out_camera_poses(
            env,
            env_cfg.viewer,
            start_eye_delta=args_cli.zoom_out_start_eye_delta,
            start_spherical=args_cli.zoom_out_start_spherical,
            start_origin_delta=args_cli.zoom_out_start_origin_delta,
            distance_delta=args_cli.zoom_out_distance_delta,
        )
        start_frames, duration_frames = args_cli.zoom_out_vid
        print(
            f"[INFO] Applying zoom-out video camera: hold {start_frames} frames, zoom for {duration_frames} frames."
        )
        print_dict(
            {
                "start_eye": zoom_camera_poses[0].tolist(),
                "start_lookat": zoom_camera_poses[1].tolist(),
                "end_eye": zoom_camera_poses[2].tolist(),
                "end_lookat": zoom_camera_poses[3].tolist(),
            },
            nesting=4,
        )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if zoom_camera_poses is not None:
                eye, lookat = _interpolate_camera_pose(
                    timestep,
                    start_frames,
                    duration_frames,
                    zoom_camera_poses[0],
                    zoom_camera_poses[1],
                    zoom_camera_poses[2],
                    zoom_camera_poses[3],
                )
                _set_video_camera_pose(env, eye, lookat)
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
