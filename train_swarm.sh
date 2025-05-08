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
# pip uninstall torch torchvision torchaudio -y

# pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

export OPENSCENE_DATA_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data/maps"
export NAVSIM_DEVKIT_ROOT="${WORKING_PATH}"
export NAVSIM_EXP_ROOT="/job_data"
export NAVSIM_CACHE_PATH="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache_HO"
# /horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/
# cache_path = "/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/"


pip install -e . -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# agent_name=vanilla_diffusiondrive_agent
# agent_name=vddrive_ho
# agent_name=vddrive_ho
agent_name=diffusiondrive_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=$agent_name \
        train_test_split=navtrain  \
        split=trainval \
        cache_path=$NAVSIM_CACHE_PATH \
        use_cache_without_dataset=True  \
        force_cache_computation=False  
        # "+agent.config.traj_norm=minmax"
        # "+agent.config.clamp=False" 

        # "dataloader.params.batch_size=128"
        # "+agent.config.clamp=False" 
        # "+agent.config.traj_norm=minmax" \
        # "+agent.config.HO_MODE=vel"
        # debug=True

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
        experiment_name=${agent_name}


# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
#         train_test_split=navtest \
#         agent=$agent_name \
#         worker=ray_distributed \
#         "agent.checkpoint_path=$escaped_path"\
#         metric_cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/metric_cache/" \
#         "+agent.config.infer_step_num=20" 
#         experiment_name=${agent_name}_eval_step20

# rm $NAVSIM_DEVKIT_ROOT/navsim/agents/$agent_name/transfuser_config.py

# mv $NAVSIM_DEVKIT_ROOT/navsim/agents/$agent_name/transfuser_config_step2.py $NAVSIM_DEVKIT_ROOT/navsim/agents/$agent_name/transfuser_config.py

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
#         train_test_split=navtest \
#         agent=$agent_name \
#         worker=ray_distributed \
#         "agent.checkpoint_path=$escaped_path"\
#         metric_cache_path="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/metric_cache/" \
#         experiment_name=${agent_name}_eval_step2


# aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir 