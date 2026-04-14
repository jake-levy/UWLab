# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_warmup_steps",
    type=int,
    default=0,
    help="Number of environment steps to run before video recording starts.",
)
parser.add_argument(
    "--video_name",
    type=str,
    default=None,
    help="Filename prefix for recorded videos. Defaults to Gymnasium's standard prefix if unset.",
)
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
    "--render_interval",
    type=int,
    default=2,
    help="Number of physics steps between captured video frames. Lower values increase visual sampling density.",
)
parser.add_argument(
    "--camera-zoom",
    type=float,
    default=1.0,
    help="Scale the viewer eye-to-lookat distance. Values < 1 zoom in, values > 1 zoom out.",
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
    "--enable_ambient_occlusion",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable ambient occlusion for rendering and recorded videos.",
)
parser.add_argument(
    "--dome_light_intensity",
    type=float,
    default=None,
    help="Override the scene dome light intensity.",
)
parser.add_argument(
    "--dome_light_hdri",
    type=str,
    default=None,
    help="Override the scene dome light HDRI. Use 'soft' or pass a direct HDRI path.",
)
parser.add_argument(
    "--hide_vention_metal",
    action="store_true",
    default=False,
    help="Hide the table's /visuals/vention_metal mesh in all environments.",
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
parser.add_argument(
    "--perturb-video",
    action="store_true",
    default=False,
    help="Enable perturbation video overlays (currently badge + frame). Requires --video.",
)
parser.add_argument(
    "--perturb-xyz",
    nargs=3,
    type=float,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Constant Cartesian perturbation added to the first three action dimensions.",
)
parser.add_argument(
    "--perturb-intervals",
    nargs=2,
    type=int,
    metavar=("OFF_FRAMES", "ON_FRAMES"),
    default=None,
    help="Cycle perturbation in recorded video frames with OFF_FRAMES disabled followed by ON_FRAMES enabled.",
)
parser.add_argument(
    "--alternate-perturb",
    action="store_true",
    default=False,
    help="Alternate the perturbation sign on successive intervals or steps.",
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
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import cv2

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.dict import print_dict
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import EE_BODY_NAME

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as omnireset_mdp
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _scale_camera_distance(eye, lookat, zoom_factor: float):
    """Scale the eye-to-lookat distance while preserving the view direction."""
    eye = np.asarray(eye, dtype=float)
    lookat = np.asarray(lookat, dtype=float)
    view_vec = eye - lookat
    view_distance = float(np.linalg.norm(view_vec))
    if view_distance <= 1e-6:
        raise ValueError("Viewer eye and lookat cannot be identical when scaling the camera distance.")
    return lookat + view_vec * zoom_factor


def _compute_zoom_out_camera_poses(
    env,
    viewer_cfg,
    start_eye_delta=None,
    start_spherical=None,
    start_origin_delta=None,
    distance_delta=0.0,
    camera_zoom=1.0,
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
    start_eye = _scale_camera_distance(start_eye, start_lookat, camera_zoom)

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


def _write_video_frame(video_writer, frame, video_width: int, video_height: int):
    """Convert a rendered RGB frame to MP4 format and write it."""
    if frame is None:
        return
    if not isinstance(frame, np.ndarray):
        return
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        return

    if frame.shape[2] == 4:
        frame = frame[..., :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
    video_writer.write(frame)


def _step_manager_env_with_render_capture(env, action, capture_callback, step_callback=None):
    """Advance a manager-based env while capturing intermediate renders."""
    env.action_manager.process_action(action.to(env.device))

    env.recorder_manager.record_pre_step()

    # Hold the same policy action across the full decimation window, but allow
    # the caller to record frames after each physics step.
    for physics_step in range(env.cfg.decimation):
        env._sim_step_counter += 1
        if step_callback is not None:
            step_callback(physics_step, env.cfg.decimation)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.recorder_manager.record_post_physics_decimation_step()
        capture_callback(physics_step, env.cfg.decimation)
        env.scene.update(dt=env.physics_dt)

    # post-step:
    # -- update env counters (used for curriculum generation)
    env.episode_length_buf += 1  # step in current episode (per env)
    env.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    # # -- reward computation
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    if len(env.recorder_manager.active_terms) > 0:
        # update observations for recording if needed
        env.obs_buf = env.observation_manager.compute()
        env.recorder_manager.record_post_step()

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        # trigger recorder terms for pre-reset calls
        env.recorder_manager.record_pre_reset(reset_env_ids)

        env._reset_idx(reset_env_ids)
        # update articulation kinematics
        env.scene.write_data_to_sim()

        # # trigger recorder terms for post-reset calls
        env.recorder_manager.record_post_reset(reset_env_ids)

    # -- update command
    env.command_manager.compute(dt=env.step_dt)
    # -- step interval events
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    env.obs_buf = env.observation_manager.compute(update_history=True)

    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


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


def _resolve_hdri_override(hdri_arg: str) -> str:
    """Resolve a CLI HDRI override into a concrete texture path."""
    hdri_presets = {
        "soft": f"{ISAAC_NUCLEUS_DIR}/Environments/Outdoor/Rivermark/dsready_content/nv_core/common_tools/content_tagging/studio_lights/Materials/photo_studio_01_4k.hdr",
        "softer": f"{ISAAC_NUCLEUS_DIR}/Assets/Skies/Cloudy/table_mountain_1_4k.hdr",
    }
    return hdri_presets.get(hdri_arg, hdri_arg)


def _resolve_dome_light_texture_file(env_cfg):
    """Resolve the dome/sky light HDRI to a local path before scene creation."""
    sky_light_cfg = getattr(env_cfg.scene, "sky_light", None) or getattr(env_cfg.scene, "dome_light", None)
    if sky_light_cfg is None or not hasattr(sky_light_cfg, "spawn") or sky_light_cfg.spawn is None:
        return

    texture_file = getattr(sky_light_cfg.spawn, "texture_file", None)
    if not texture_file:
        return

    try:
        sky_light_cfg.spawn.texture_file = retrieve_file_path(texture_file)
    except FileNotFoundError:
        # Fall back to the configured texture path when the local cache does not have the file.
        # This keeps offline/headless test environments from failing during launch.
        print(f"[WARN] Could not resolve dome light texture locally: {texture_file}")


def _log_render_settings(env_cfg):
    """Print the effective render and dome light settings after CLI overrides."""
    sky_light_cfg = getattr(env_cfg.scene, "sky_light", None) or getattr(env_cfg.scene, "dome_light", None)
    render_settings = {
        "ambient_occlusion": getattr(env_cfg.sim.render, "enable_ambient_occlusion", None),
        "reflections": getattr(env_cfg.sim.render, "enable_reflections", None),
        "dlssg": getattr(env_cfg.sim.render, "enable_dlssg", None),
        "dl_denoiser": getattr(env_cfg.sim.render, "enable_dl_denoiser", None),
        "dome_light_intensity": None,
        "dome_light_hdri": None,
    }
    if sky_light_cfg is not None and hasattr(sky_light_cfg, "spawn") and sky_light_cfg.spawn is not None:
        render_settings["dome_light_intensity"] = getattr(sky_light_cfg.spawn, "intensity", None)
        render_settings["dome_light_hdri"] = getattr(sky_light_cfg.spawn, "texture_file", None)

    print("[INFO] Effective render settings:")
    print_dict(render_settings, nesting=4)


def _log_stage_dome_light_settings():
    """Print the live USD dome/sky light settings after scene creation."""
    stage = sim_utils.get_current_stage()
    light_prim = stage.GetPrimAtPath("/World/skyLight")
    light_path = "/World/skyLight"
    if not light_prim.IsValid():
        light_prim = stage.GetPrimAtPath("/World/domeLight")
        light_path = "/World/domeLight"
    if not light_prim.IsValid():
        print("[INFO] Live stage dome light not found at /World/skyLight or /World/domeLight.")
        return

    stage_settings = {
        "prim_path": light_path,
        "inputs:intensity": None,
        "inputs:exposure": None,
        "inputs:texture:file": None,
        "inputs:texture:format": None,
        "visibleInPrimaryRay": None,
    }
    for attr_name in stage_settings:
        if attr_name == "prim_path":
            continue
        attr = light_prim.GetAttribute(attr_name)
        if attr.IsValid():
            stage_settings[attr_name] = attr.Get()

    print("[INFO] Live stage dome light settings:")
    print_dict(stage_settings, nesting=4)


def _hide_vention_metal_visuals():
    """Hide the pat_vention metal visuals in all cloned environments."""
    prim_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/Table/visuals/vention_metal")
    if not prim_paths:
        raise ValueError(
            "--hide_vention_metal could not find any prims matching "
            "'/World/envs/env_.*/Table/visuals/vention_metal'."
        )

    for prim_path in prim_paths:
        prim = sim_utils.get_prim_at_path(prim_path)
        sim_utils.set_prim_visibility(prim, False)

    print(f"[INFO] Hid {len(prim_paths)} vention_metal visual prim(s).")


@dataclass
class PerturbationOverlayState:
    """Shared perturbation state used by the video overlay wrapper."""

    active: bool = False
    delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PerturbationArrowRenderer:
    """Render a world-space arrow near the controlled wrist using VisualizationMarkers."""

    def __init__(self, env, body_name: str, env_index: int = 0):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

        self._env = env
        self._robot = env.unwrapped.scene["robot"]
        self._env_index = env_index
        body_ids, body_names = self._robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one match for perturbation arrow body '{body_name}', got {body_names}.")
        self._body_idx = body_ids[0]
        self._tip_offset = 0.004
        self._body_scale = 0.045
        self._length_scale = 0.35 / 20.0
        self._min_arrow_length = 5e-5

        marker_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Perturbation/arrow")
        marker_cfg.markers["arrow"].scale = (1.0, 1.0, 1.0)
        if hasattr(marker_cfg.markers["arrow"], "visual_material"):
            marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.95, 0.15, 0.12)
        self._marker = VisualizationMarkers(marker_cfg)
        self._marker.set_visibility(False)

    def clear(self):
        self._marker.set_visibility(False)

    @staticmethod
    def _quat_from_x_axis(direction: torch.Tensor) -> torch.Tensor:
        """Return quaternion that rotates +X to the given unit direction."""
        x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
        dot = torch.dot(x_axis, direction)

        if dot < -0.999999:
            return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=direction.dtype, device=direction.device)

        xyz = torch.cross(x_axis, direction, dim=0)
        quat = torch.cat([torch.tensor([1.0 + dot], dtype=direction.dtype, device=direction.device), xyz], dim=0)
        return quat / torch.linalg.vector_norm(quat)

    def update(self, perturb_xyz_base: torch.Tensor | None, active: bool):
        if not active or perturb_xyz_base is None:
            self.clear()
            return

        perturb_norm = torch.linalg.vector_norm(perturb_xyz_base).item()
        if perturb_norm <= 1e-8:
            self.clear()
            return

        env_idx = self._env_index
        ee_pos_w = self._robot.data.body_pos_w[env_idx, self._body_idx]
        root_quat_w = self._robot.data.root_quat_w[env_idx]
        perturb_world = math_utils.quat_apply(root_quat_w.unsqueeze(0), perturb_xyz_base.unsqueeze(0)).squeeze(0)
        direction = perturb_world / torch.linalg.vector_norm(perturb_world)
        tip = ee_pos_w + direction * self._tip_offset
        arrow_length = max(perturb_norm * self._length_scale, self._min_arrow_length)
        tail = tip - direction * arrow_length
        orientation = self._quat_from_x_axis(direction).unsqueeze(0)
        translation = tail.unsqueeze(0)
        scale = torch.tensor(
            [[arrow_length, self._body_scale, self._body_scale]],
            dtype=perturb_world.dtype,
            device=perturb_world.device,
        )
        self._marker.set_visibility(True)
        self._marker.visualize(translations=translation, orientations=orientation, scales=scale)


class PerturbationVideoOverlayWrapper(gym.Wrapper):
    """Draw simple perturbation overlays onto rendered RGB frames."""

    def __init__(self, env: gym.Env, overlay_state: PerturbationOverlayState):
        super().__init__(env)
        self._overlay_state = overlay_state

    def render(self):
        frame = self.env.render()
        if frame is None or not self._overlay_state.active:
            return frame
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        border_px = max(8, min(height, width) // 45)
        color = (235, 64, 52)

        cv2.rectangle(annotated, (0, 0), (width - 1, height - 1), color, border_px)

        badge_w = min(width - 24, max(280, width // 4))
        badge_h = max(72, height // 14)
        badge_x, badge_y = 18, 18
        overlay = annotated.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), color, thickness=-1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0.0, annotated)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = max(0.8, width / 1700.0)
        cv2.putText(
            annotated,
            "PERTURBATION ACTIVE",
            (badge_x + 16, badge_y + 30),
            font,
            title_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated


def _is_perturbation_active(frame_idx: int, intervals: tuple[int, int] | None) -> bool:
    """Return whether perturbation should be active on the given frame."""
    if intervals is None:
        return True

    off_frames, on_frames = intervals
    cycle_frames = off_frames + on_frames
    if cycle_frames <= 0:
        return False
    return (frame_idx % cycle_frames) >= off_frames


def _perturbation_sign(frame_idx: int, intervals: tuple[int, int] | None, alternate: bool) -> float:
    """Return the perturbation sign multiplier for the given frame."""
    if not alternate:
        return 1.0
    if intervals is None:
        return -1.0 if frame_idx % 2 else 1.0

    off_frames, on_frames = intervals
    cycle_frames = off_frames + on_frames
    if cycle_frames <= 0:
        return 1.0
    if off_frames == 0:
        return -1.0 if ((frame_idx // on_frames) % 2 == 1) else 1.0
    return -1.0 if ((frame_idx // cycle_frames) % 2 == 1) else 1.0


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
    if args_cli.enable_ambient_occlusion is not None:
        env_cfg.sim.render.enable_ambient_occlusion = args_cli.enable_ambient_occlusion
    if args_cli.dome_light_intensity is not None:
        if args_cli.dome_light_intensity < 0.0:
            raise ValueError("--dome_light_intensity must be >= 0.")
        sky_light_cfg = getattr(env_cfg.scene, "sky_light", None) or getattr(env_cfg.scene, "dome_light", None)
        if sky_light_cfg is None or not hasattr(sky_light_cfg, "spawn") or sky_light_cfg.spawn is None:
            raise ValueError("This task does not expose a configurable dome/sky light.")
        if not hasattr(sky_light_cfg.spawn, "intensity"):
            raise ValueError("This task's sky light does not support intensity override.")
        sky_light_cfg.spawn.intensity = args_cli.dome_light_intensity
    if args_cli.dome_light_hdri is not None:
        sky_light_cfg = getattr(env_cfg.scene, "sky_light", None) or getattr(env_cfg.scene, "dome_light", None)
        if sky_light_cfg is None or not hasattr(sky_light_cfg, "spawn") or sky_light_cfg.spawn is None:
            raise ValueError("This task does not expose a configurable dome/sky light.")
        if not hasattr(sky_light_cfg.spawn, "texture_file"):
            raise ValueError("This task's sky light does not support HDRI override.")
        sky_light_cfg.spawn.texture_file = _resolve_hdri_override(args_cli.dome_light_hdri)

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
    if args_cli.video_warmup_steps < 0:
        raise ValueError("--video_warmup_steps must be >= 0.")
    if args_cli.video_name is not None:
        if not args_cli.video:
            raise ValueError("--video_name requires --video.")
        if not args_cli.video_name.strip():
            raise ValueError("--video_name must not be empty.")
    if args_cli.video_size is not None:
        if not args_cli.video:
            raise ValueError("--video_size requires --video.")
        if args_cli.video_size[0] <= 0 or args_cli.video_size[1] <= 0:
            raise ValueError("WIDTH and HEIGHT for --video_size must be > 0.")

        env_cfg.viewer.resolution = (args_cli.video_size[0], args_cli.video_size[1])
    if args_cli.render_interval <= 0:
        raise ValueError("--render_interval must be > 0.")
    if args_cli.camera_zoom <= 0.0:
        raise ValueError("--camera-zoom must be > 0.")
    env_cfg.sim.render_interval = args_cli.render_interval
    if args_cli.perturb_video and not args_cli.video:
        raise ValueError("--perturb-video requires --video.")
    if args_cli.perturb_intervals is not None:
        off_frames, on_frames = args_cli.perturb_intervals
        if off_frames < 0:
            raise ValueError("OFF_FRAMES for --perturb-intervals must be >= 0.")
        if on_frames <= 0:
            raise ValueError("ON_FRAMES for --perturb-intervals must be > 0.")

    perturb_xyz = None
    perturb_intervals = tuple(args_cli.perturb_intervals) if args_cli.perturb_intervals is not None else None
    perturb_overlay_state = PerturbationOverlayState()
    if args_cli.perturb_xyz is not None:
        perturb_xyz = torch.tensor(args_cli.perturb_xyz, dtype=torch.float32)
        perturb_overlay_state.delta_xyz = tuple(float(value) for value in args_cli.perturb_xyz)
        perturb_overlay_state.active = bool(
            torch.linalg.vector_norm(perturb_xyz).item() > 0.0 and _is_perturbation_active(0, perturb_intervals)
        )

    _resolve_dome_light_texture_file(env_cfg)
    _log_render_settings(env_cfg)

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
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _log_stage_dome_light_settings()
    if args_cli.hide_vention_metal:
        _hide_vention_metal_visuals()

    perturb_arrow_renderer = None
    if args_cli.perturb_video:
        viewer_env_index = max(0, int(getattr(env_cfg.viewer, "env_index", 0)))
        perturb_arrow_renderer = PerturbationArrowRenderer(base_env, body_name=EE_BODY_NAME, env_index=viewer_env_index)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    if args_cli.perturb_video:
        base_env = PerturbationVideoOverlayWrapper(base_env, perturb_overlay_state)

    rl_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    video_writer = None
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_folder, exist_ok=True)
        video_width, video_height = (
            tuple(args_cli.video_size) if args_cli.video_size is not None else tuple(env_cfg.viewer.resolution)
        )
        video_fps = args_cli.video_fps if args_cli.video_fps is not None else int(round(1.0 / env_cfg.sim.dt))
        if video_fps <= 0:
            raise ValueError("Computed video fps must be > 0.")
        video_name = args_cli.video_name if args_cli.video_name is not None else "play"
        video_path = os.path.join(video_folder, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (video_width, video_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer at {video_path}.")
        print("[INFO] Recording videos during training.")
        print_dict(
            {
                "video_folder": video_folder,
                "video_path": video_path,
                "video_length": args_cli.video_length,
                "fps": video_fps,
                "render_interval": args_cli.render_interval,
            },
            nesting=4,
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=base_env.unwrapped.device)

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

    step_env = base_env.unwrapped
    dt = step_env.step_dt
    if perturb_xyz is not None:
        perturb_xyz = perturb_xyz.to(step_env.device)

    # reset environment
    obs = rl_env.get_observations()
    policy_timestep = 0
    video_frame_timestep = 0
    zoom_camera_poses = None
    default_video_camera_pose = None
    if args_cli.video:
        default_video_camera_pose = (
            _scale_camera_distance(env_cfg.viewer.eye, env_cfg.viewer.lookat, args_cli.camera_zoom),
            np.asarray(env_cfg.viewer.lookat, dtype=float),
        )
    if args_cli.zoom_out_vid is not None:
        zoom_camera_poses = _compute_zoom_out_camera_poses(
            step_env,
            env_cfg.viewer,
            start_eye_delta=args_cli.zoom_out_start_eye_delta,
            start_spherical=args_cli.zoom_out_start_spherical,
            start_origin_delta=args_cli.zoom_out_start_origin_delta,
            distance_delta=args_cli.zoom_out_distance_delta,
            camera_zoom=args_cli.camera_zoom,
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
    elif default_video_camera_pose is not None:
        print(f"[INFO] Applying video camera zoom factor: {args_cli.camera_zoom}")
        print_dict(
            {
                "eye": default_video_camera_pose[0].tolist(),
                "lookat": default_video_camera_pose[1].tolist(),
            },
            nesting=4,
        )
        _set_video_camera_pose(base_env, default_video_camera_pose[0], default_video_camera_pose[1])
    if perturb_xyz is not None:
        print(f"[INFO] Applying constant Cartesian perturbation: {tuple(float(v) for v in args_cli.perturb_xyz)}")
    if args_cli.alternate_perturb:
        print("[INFO] Alternating perturbation sign enabled (p, -p, p, -p, ...).")
    if perturb_intervals is not None:
        print(
            f"[INFO] Applying perturbation intervals: off for {perturb_intervals[0]} frame(s), "
            f"on for {perturb_intervals[1]} frame(s)."
        )
    if args_cli.perturb_video:
        print("[INFO] Perturbation video overlays enabled (badge + frame + arrow).")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if zoom_camera_poses is not None:
                video_timestep = max(0, policy_timestep - args_cli.video_warmup_steps)
                eye, lookat = _interpolate_camera_pose(
                    video_timestep,
                    start_frames,
                    duration_frames,
                    zoom_camera_poses[0],
                    zoom_camera_poses[1],
                    zoom_camera_poses[2],
                    zoom_camera_poses[3],
                )
                _set_video_camera_pose(base_env, eye, lookat)
            # agent stepping
            actions = policy(obs)
            if video_writer is not None:
                # Manual stepping keeps the policy action fixed while allowing us to capture
                # multiple physics renders between policy updates.
                current_perturb_active = perturb_xyz is not None and _is_perturbation_active(
                    video_frame_timestep, perturb_intervals
                )
                current_perturb_sign = (
                    _perturbation_sign(video_frame_timestep, perturb_intervals, args_cli.alternate_perturb)
                    if current_perturb_active
                    else 1.0
                )
                step_action = actions.clone()
                if current_perturb_active:
                    if actions.shape[-1] < 3:
                        raise ValueError("--perturb-xyz requires the policy action space to have at least 3 dimensions.")
                    step_action[:, :3] += perturb_xyz * current_perturb_sign

                def step_callback(physics_step_idx: int, total_physics_steps: int):
                    nonlocal current_perturb_active, current_perturb_sign
                    step_perturb_active = perturb_xyz is not None and _is_perturbation_active(
                        video_frame_timestep, perturb_intervals
                    )
                    step_perturb_sign = (
                        _perturbation_sign(video_frame_timestep, perturb_intervals, args_cli.alternate_perturb)
                        if step_perturb_active
                        else 1.0
                    )
                    if step_perturb_active != current_perturb_active or step_perturb_sign != current_perturb_sign:
                        if actions.shape[-1] < 3:
                            raise ValueError(
                                "--perturb-xyz requires the policy action space to have at least 3 dimensions."
                            )
                        step_action = actions.clone()
                        if step_perturb_active:
                            step_action[:, :3] += perturb_xyz * step_perturb_sign
                        env_action = step_action.to(step_env.device)
                        step_env.action_manager.process_action(env_action)
                        current_perturb_active = step_perturb_active
                        current_perturb_sign = step_perturb_sign

                    perturb_overlay_state.active = step_perturb_active
                    if perturb_arrow_renderer is not None:
                        step_perturb_xyz = perturb_xyz * step_perturb_sign if step_perturb_active else None
                        perturb_arrow_renderer.update(step_perturb_xyz, step_perturb_active)

                def capture_callback(physics_step_idx: int, total_physics_steps: int):
                    nonlocal video_frame_timestep
                    should_capture = ((physics_step_idx + 1) % args_cli.render_interval == 0) or (
                        physics_step_idx == total_physics_steps - 1
                    )
                    if not should_capture:
                        return
                    frame = base_env.render()
                    _write_video_frame(video_writer, frame, video_width, video_height)
                    video_frame_timestep += 1

                _, _, terminated, truncated, _ = _step_manager_env_with_render_capture(
                    step_env, step_action, capture_callback, step_callback=step_callback
                )
                dones = (terminated | truncated).to(dtype=torch.long)
                obs = rl_env.get_observations()
            else:
                perturb_active = perturb_xyz is not None and _is_perturbation_active(policy_timestep, perturb_intervals)
                if perturb_active:
                    if actions.shape[-1] < 3:
                        raise ValueError("--perturb-xyz requires the policy action space to have at least 3 dimensions.")
                    perturb_sign = _perturbation_sign(policy_timestep, perturb_intervals, args_cli.alternate_perturb)
                    actions[:, :3] += perturb_xyz * perturb_sign
                # env stepping
                obs, _, dones, _ = rl_env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            policy_timestep += 1
            # Exit the play loop after recording one video
            if policy_timestep == args_cli.video_warmup_steps + args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if perturb_arrow_renderer is not None:
        perturb_arrow_renderer.clear()
    if video_writer is not None:
        video_writer.release()
    rl_env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
