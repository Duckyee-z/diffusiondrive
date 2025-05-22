agent_name=vddrivev2.3
escaped_path="/home/users/zhiyu.zheng/workplace/e2ead/vdd/diffusiondrive/tb_logs/speed_anchor_embed.ckpt"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/metric_cache/" \
        experiment_name=${agent_name}_test_exp \
        +agent.config.infer_step_num=2 \
        +agent.config.anchor_embed=True\
        +agent.config.with_query_as_embedding=True \
        +agent.config.infer_timestep_spacing=leading 
        
