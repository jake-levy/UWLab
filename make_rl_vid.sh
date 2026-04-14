#!/bin/bash

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 25 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 600 \
    --camera-zoom 1.5 \
    --video_fps 30 \
    --video_size 1920 1080 \
    --video_name rl \
    --render_interval 2 \
    --autoreset \
    --dome_light_hdri soft \
    --disable-dlssg \
    --hide_vention_metal \
    --camera-lookat-delta 2.5 0.0 0.0 \
    --ground-z-delta 0.2



    # --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    # --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
