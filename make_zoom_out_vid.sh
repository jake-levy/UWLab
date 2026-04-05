python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 10 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 70 \
    --autoreset \
    --zoom-out-vid 30 10
