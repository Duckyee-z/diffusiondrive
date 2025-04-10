#!/bin/bash
# ${WORKING_PATH} :/running_package/job
cp /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/miniconda.tar.gz ${WORKING_PATH}
cp /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/02_envs/navsim.tar.gz ${WORKING_PATH}
cd ${WORKING_PATH}
tar -zxf miniconda.tar.gz -C /
mkdir -p /home/users/zhiyu.zheng/miniconda/envs/navsim
tar -zxf navsim.tar.gz -C /home/users/zhiyu.zheng/miniconda/envs/navsim
rm -f miniconda.tar.gz navsim.tar.gz

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
export NAVSIM_CACHE_PATH = "/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/"
# /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/
# cache_path = "/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/"


pip install -e . -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# agent_name=vanilla_diffusiondrive_agent
agent_name=vdiffusiondrivev2


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=${agent_name} \
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 

ckpt_path=$(find $NAVSIM_EXP_ROOT/ -type f -name '*.ckpt')
# echo $ckpt_path
escaped_path=${ckpt_path//=/\\=}
# echo $escaped_path
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/metric_cache/" \
        experiment_name=${agent_name}_eval


# aidi-inf-cli job submit -f EDA.yaml -t ~/temp_dir