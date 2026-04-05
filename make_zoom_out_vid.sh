# python scripts/reinforcement_learning/rsl_rl/play.py \
#     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
#     --num_envs 81 \
#     --checkpoint peg_state_rl_expert_seed42.pt \
#     env.scene.insertive_object=peg \
#     env.scene.receptive_object=peghole \
#     --headless \
#     --video \
#     --video_length 105 \
#     --autoreset \
#     --zoom-out-vid 45 15 \
#     --zoom-out-start-spherical 0 60 2.0 \
#     --zoom-out-distance-delta -8.0 \
#     --video_fps 30 \
#     --video_size 1920 1080

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 81 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 35 \
    --autoreset \
    --zoom-out-vid 20 5 \
    --zoom-out-start-spherical 0 60 1.2 \
    --zoom-out-start-origin-delta 0.3 0.0 0.0 \
    --zoom-out-distance-delta -6.0 \
    --video_fps 30 \
    --video_size 1920 1080 \
    --dome_light_hdri soft




    # --disable_ambient_occlusion