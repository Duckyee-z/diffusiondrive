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