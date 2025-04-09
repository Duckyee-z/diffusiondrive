
agent_name=vanilla_diffusiondrive_agent

CUDA_VISIBLE_DEVICES=6 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=$agent_name\
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        'trainer.params.max_epochs=2' \
        'debug=true'

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

