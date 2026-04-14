#    --dome_light_hdri soft \
    # --dome_light_intensity 1000
CAMERA_ZOOM="${CAMERA_ZOOM:-1.0}"
        
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 81 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --video \
    --video_length 500 \
    --camera-zoom "${CAMERA_ZOOM}" \
    --video_fps 30 \
    --video_size 1920 1080 \
    --video_name zoom_out_rev2 \
    --render_interval 2 \
    --autoreset \
    --zoom-out-vid 200 80 \
    --zoom-out-start-spherical 0 40 1.2 \
    --zoom-out-start-origin-delta 0.2 0.0 0.0 \
    --zoom-out-distance-delta -3.0 \
    --enable_ambient_occlusion \
    --dome_light_hdri soft \
    --video_warmup_steps 0 \
    --hide_vention_metal \
    --disable-dlssg \
    --ground-z-delta 0.2 \
    --dome_light_intensity 700
