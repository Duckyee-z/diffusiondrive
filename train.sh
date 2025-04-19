agent_name=vdiffusiondrivev2

CUDA_VISIBLE_DEVICES=6 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_myvisual.py \
        agent=$agent_name \
        experiment_name=$agent_name\
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=2 \
        debug=true

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

