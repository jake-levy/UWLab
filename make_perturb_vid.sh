DEVICE_ARGS=()
if [[ -n "${UWLAB_DEVICE:-}" ]]; then
    DEVICE_ARGS+=(--device "${UWLAB_DEVICE}")
fi
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"

python scripts/reinforcement_learning/rsl_rl/play.py \
    "${DEVICE_ARGS[@]}" \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 1 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length "${VIDEO_LENGTH}" \
    --video_fps 30 \
    --video_size 1920 1080 \
    --video_name perturb_final \
    --render_interval 2 \
    --perturb-video \
    --perturb-xyz 0.00 25.0 0.0 \
    --alternate-perturb \
    --perturb-intervals 80 45 \
    --autoreset \
    --dome_light_hdri soft


    # --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    # --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \