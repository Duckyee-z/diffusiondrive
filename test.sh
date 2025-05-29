agent_name=vddrivev2.3
escaped_path="/home/users/zhiyu.zheng/workplace/e2ead/vdd/diffusiondrive/tb_logs/Speedanchorv4-norm-5to5-NOclamp.ckpt"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/metric_cache/" \
        experiment_name=${agent_name}_test_exp \
        +agent.config.anchor_embed=True\
        +agent.config.with_query_as_embedding=True \
        +agent.config.norm_scale=5\
        +agent.config.use_clamp=False

        # +agent.config.infer_step_num=10
        # +agent.config.with_query_as_embedding=True \
        # +agent.config.infer_timestep_spacing=leading 
        # +agent.config.infer_step_num=4 \
        
