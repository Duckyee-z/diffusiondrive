
# agent_name=vddrivev2
# agent_name=vdiffusiondrivev2_minmaxnorm
# agent_name=vddrivev2.3



agent_name=speedanchorv4.3
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=3 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
#         agent=$agent_name \
#         experiment_name=test_exp \
#         train_test_split=navtrain  \
#         split=trainval   \
#         cache_path="/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/training_cache/" \
#         use_cache_without_dataset=True  \
#         force_cache_computation=False \
#         +agent.config.norm_scale=1\
#         +agent.config.random_scale=1.0\
#         +agent.config.use_clamp=True \
#         +agent.config.use_manual_timesteps=True\
#         +agent.config.manual_timesteps=\'750,250\' \
#         debug=True

        # +agent.config.use_different_loss_weight=True 
        # +agent.config.trajectory_weight=10
        # +agent.config.use_mse_loss=True \
        # +agent.config.odo_loss=True\
        # debug=true
        

        # 'trainer.params.max_epochs=15' \
        # +agent.config.use_mse_loss=True 
        # +agent.config.anchor_embed=True \
        # +agent.config.anchor_embed_interact=True

        # +agent.config.with_query_as_embedding=True
        # +agent.config.anchor_embed_interact=True

# CUDA_VISIBLE_DEVICES=6 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
#         agent=$agent_name \
#         experiment_name=$agent_name\
#         train_test_split=navtrain  \
#         split=trainval   \
#         cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/" \
#         use_cache_without_dataset=True  \
#         force_cache_computation=False \
#         'trainer.params.max_epochs=2' \
#         'debug=true'

# ckpt_path=$(find $NAVSIM_EXP_ROOT/$agent_name -type f -name '*.ckpt')
# echo $ckpt_path
# escaped_path=${ckpt_path//=/\\=}
# echo $escaped_path
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
#         train_test_split=navtest \
#         agent=$agent_name \
#         worker=ray_distributed \
#         "agent.checkpoint_path=$escaped_path"\
#         experiment_name=${agent_name}_eval

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_data_analyse_ho.py \
        agent=$agent_name \
        experiment_name=$agent_name\
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=2 \
        cache_path=/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/training_cache \
        use_cache_without_dataset=True \
        force_cache_computation=False 
        # debug=True 


# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_myvisual.py \
#         agent=$agent_name \
#         experiment_name=$agent_name\
#         train_test_split=navtrain  \
#         split=trainval   \
#         trainer.params.max_epochs=2 \
#         cache_path=/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/training_cache \
#         use_cache_without_dataset=True \
#         force_cache_computation=False 
        # debug=True 