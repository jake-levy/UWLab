python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 1 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 180 \
    --video_fps 30 \
    --video_size 1920 1080 \
    --video_name perturb \
    --perturb-video \
    --perturb-xyz 0.00 1.0 0.0 \
    --perturb-intervals 30 30 \
    --dome_light_hdri soft
