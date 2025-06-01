# 数据位置
export OPENSCENE_DATA_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/navsim_data/maps"
export NAVSIM_DEVKIT_ROOT="${WORKING_PATH}" # 代码位置
export NAVSIM_EXP_ROOT="/job_data" # 实验结果保存位置
export NAVSIM_CACHE_PATH = "/horizon-bucket/saturn_v_dev/01_users/zhiyu.zheng/01_dataset/01_E2EAD/01_nuscenes/exp/training_cache/"

pip install -e . -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple --extra-index-url=http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

agent_name=speedanchorv4.1
# train
# 默认配置在navsim/agents/speedanchorv4_1/transfuser_config.py 下面
CUDA_VISIBLE_DEVICES=2 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=$agent_name \
        experiment_name=test_exp \
        train_test_split=navtrain  \
        split=trainval   \
        cache_path="/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +agent.config.norm_scale=1\
        +agent.config.use_clamp=True\
        +agent.config.output_result=trajectory_500
        # debug=true

# test
escaped_path="/home/users/zhiyu.zheng/workplace/e2ead/vdd/diffusiondrive/tb_logs/Speedanchorv4-norm-5to5-NOclamp.ckpt"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent_name \
        worker=ray_distributed \
        "agent.checkpoint_path=$escaped_path"\
        metric_cache_path="/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/metric_cache/" \
        experiment_name=${agent_name}_test_exp \
        +agent.config.norm_scale=1\
        +agent.config.use_clamp=True