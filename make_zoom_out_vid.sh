#    --dome_light_hdri soft \

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 121 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 138 \
    --video_fps 30 \
    --video_size 1920 1080 \
    --video_name zoom_out \
    --autoreset \
    --zoom-out-vid 60 18 \
    --zoom-out-start-spherical 0 50 1.2 \
    --zoom-out-start-origin-delta 0.3 0.0 0.0 \
    --zoom-out-distance-delta -3.0 \
    --hide_vention_metal
