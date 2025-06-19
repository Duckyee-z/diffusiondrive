agent_name=speedanchorv4.4
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=3 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=test_exp \
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +agent.config.norm_scale=1\
        +agent.config.random_scale=1.0\
        +agent.config.use_clamp=True \
        debug=True\
        +lr=1e-3