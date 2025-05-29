#!/bin/bash
# ${WORKING_PATH} :/running_package/job
cp /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/miniconda.tar.gz ${WORKING_PATH}
cp /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/02_envs/navsim.tar.gz ${WORKING_PATH}
# cp /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/training_cache.tar.gz ${WORKING_PATH}
rsync -avh --progress --inplace /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/training_cache.tar.gz ${WORKING_PATH}
cd ${WORKING_PATH}
# ls -ah
tar -zxf miniconda.tar.gz -C /
mkdir -p /home/users/zhiyu.zheng/miniconda/envs/navsim
tar -zxf navsim.tar.gz -C /home/users/zhiyu.zheng/miniconda/envs/navsim
tar -zxf training_cache.tar.gz -C ${WORKING_PATH}
rm -f miniconda.tar.gz navsim.tar.gz training_cache.tar.gz

export PATH=${WORKING_PATH}/home/users/zhiyu.zheng/miniconda/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /home/users/zhiyu.zheng/miniconda/bin/activate
# pwd
echo ======================
conda env list 
conda activate navsim
unset LD_LIBRARY_PATH

# pip install -e . -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

export OPENSCENE_DATA_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data/maps"
export NAVSIM_DEVKIT_ROOT="${WORKING_PATH}"
export NAVSIM_EXP_ROOT="/job_data"
# export NAVSIM_CACHE_PATH x= "/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/"
# /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/
pip install -e . -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# agent_name=vanilla_diffusiondrive_agent
agent_name=vddrivev2.3
random_scale='0.2'

HYDRA_FULL_ERROR=1 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=${agent_name} \
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="${WORKING_PATH}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +agent.config.anchor_embed=True \
        +agent.config.with_query_as_embedding=True \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=1\
        +agent.config.use_clamp=True \
        +agent.config.truncated_vx=True 
        # +agent.config.output_result
        # +lr=1.2e-4
        # +agent.config.trajectory_weight=5 \


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
        metric_cache_path="${WORKING_PATH}/metric_cache/" \
        experiment_name=${agent_name}_eval_500 \
        +agent.config.anchor_embed=True \
        +agent.config.with_query_as_embedding=True \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=1\
        +agent.config.use_clamp=True\
        +agent.config.truncated_vx=True \
        +agent.config.output_result=trajectory_500

    sleep 2s

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="${WORKING_PATH}/metric_cache/" \
        experiment_name=${agent_name}_eval_0 \
        +agent.config.infer_step_num=5 \
        +agent.config.anchor_embed=True \
        +agent.config.with_query_as_embedding=True \
        +agent.config.random_scale=${random_scale}\
        +agent.config.norm_scale=1\
        +agent.config.use_clamp=True\
        +agent.config.truncated_vx=True \
        +agent.config.output_result=trajectory_0
        

    sleep 2s

done

python mean_csv.py --directory /job_data/${agent_name}_eval_500/ --output /job_data/ --name step2_500
python mean_csv.py --directory /job_data/${agent_name}_eval_0/ --output /job_data/ --name step2_0
# python mean_csv.py --directory /job_data/${agent_name}_eval_step10/ --output /job_data/ --name step10


