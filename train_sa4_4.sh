#!/bin/bash
source scripts/01_env.sh
cd ${WORKING_PATH}
agent_name=speedanchorv4.4
random_scale='1.0'
norm_scale=1
num_train_timesteps=1000
num_train_timesteps_used=1000
truncated_vx=True
num_gpus=$(nvidia-smi -L | wc -l)
ray_worker=$((num_gpus * 13))
# echo $ray_worker

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=${agent_name} \
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="${WORKING_PATH}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=${norm_scale}\
        +agent.config.use_clamp=True\
        +agent.config.num_train_timesteps=$num_train_timesteps \
        +agent.config.num_train_timesteps_used=$num_train_timesteps_used\
        +agent.config.truncated_vx=${truncated_vx} \
        +agent.config.use_different_loss_weight=True \
        +agent.config.trajectory_weight=6

        # +agent.config.odo_loss=True \
        # +agent.config.use_mse_loss=True \
        # +agent.config.add_status_coding_to_condition=True 
        # +agent.config.output_result=trajectory_500
        # +agent.config.use_mse_loss=True \
        # +lr=1.2e-4
        # +agent.config.trajectory_weight=5 \
        # +agent.config.trajectory_weight=1200\



# ckpt_paths=$(find "$NAVSIM_EXP_ROOT/" -type f -name "*.ckpt")
rsync -avh --progress --inplace /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/metric_cache.tar.gz ${WORKING_PATH}
tar -zxf metric_cache.tar.gz -C ${WORKING_PATH}


ckpt_paths=()
while IFS= read -r -d $'\0' file; do
    ckpt_paths+=("$file")
done < <(find "$NAVSIM_EXP_ROOT/" -type f -name "*.ckpt" -print0)
# 打印所有找到的ckpt路径
printf '%s\n' "${ckpt_paths[@]}"

# 遍历所有找到的 .ckpt 文件
for ckpt_path in "${ckpt_paths[@]}"; do
    # 转义路径中的特殊字符（特别是 =）
    escaped_path=${ckpt_path//=/\\=}
    echo "验证: $escaped_path"

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        worker.threads_per_node=$ray_worker\
        metric_cache_path="${WORKING_PATH}/metric_cache/" \
        experiment_name=${agent_name}_eval2 \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=${norm_scale}\
        +agent.config.use_clamp=True \
        +agent.config.num_train_timesteps=$num_train_timesteps \
        +agent.config.num_train_timesteps_used=$num_train_timesteps_used\
        +agent.config.truncated_vx=${truncated_vx} 

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        worker.threads_per_node=$ray_worker\
        metric_cache_path="${WORKING_PATH}/metric_cache/" \
        experiment_name=${agent_name}_eval2_500 \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=${norm_scale}\
        +agent.config.use_clamp=True\
        +agent.config.output_result=trajectory_500 

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="${WORKING_PATH}/metric_cache/" \
        worker.threads_per_node=$ray_worker\
        experiment_name=${agent_name}_eval_step4 \
        +agent.config.infer_step_num=4 \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=${norm_scale}\
        +agent.config.use_clamp=True\
        +agent.config.num_train_timesteps=$num_train_timesteps \
        +agent.config.num_train_timesteps_used=$num_train_timesteps_used\
        +agent.config.truncated_vx=${truncated_vx} 


    # python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    #     train_test_split=navtest \
    #     agent=$agent_name \
    #     worker=ray_distributed \
    #     "agent.checkpoint_path=$escaped_path"\
    #     metric_cache_path="${WORKING_PATH}/metric_cache/" \
    #     experiment_name=${agent_name}_eval2_0 \
    #     +agent.config.random_scale=${random_scale}\
    #     +agent.config.norm_scale=1\
    #     +agent.config.use_clamp=True\
    #     +agent.config.output_result=trajectory_0 

        
    sleep 2s


done

python mean_csv.py --directory /job_data/${agent_name}_eval2/ --output /job_data/ --name step2
python mean_csv.py --directory /job_data/${agent_name}_eval2_500/ --output /job_data/ --name step2_500
# python mean_csv.py --directory /job_data/${agent_name}_eval2_0/ --output /job_data/ --name step2_0
python mean_csv.py --directory /job_data/${agent_name}_eval_step4/ --output /job_data/ --name step4


