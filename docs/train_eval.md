# DiffusionDrive Training and Evaluation

## 1. Cache dataset for faster training and evaluation
```bash
# cache dataset for training
python navsim/planning/script/run_dataset_caching.py agent=vdiffusiondrivev2 experiment_name=vdiffusiondrivev2 train_test_split=navtrain

# cache dataset for evaluation
python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

## 2. Training
If your training machine does not have network access, you should download the pretrained ResNet-34 model from [huggingface](https://huggingface.co/timm/resnet34.a1_in1k) and upload it to your training machine. You should also download the [clustered anchors](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy)

Before starting training, ensure that you correctly set the `bkb_path` to the path of the downloaded pretrained ResNet-34 model. Additionally, set the `plan_anchor_path` to the path of the downloaded clustered anchors in the file located at `/path/to/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py`.

```bash
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="1,2,3,4" python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=diffusiondrive_agent \
        experiment_name=vanilla_diffusiondrive_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 
```
CUDA_VISIBLE_DEVICES="1,2,3,4" python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent1 train_test_split=navtrain split=trainval cache_path="${NAVSIM_EXP_ROOT}/training_cache/use_cache_without_dataset=True force_cache_computation=False trainer.params.max_epochs=100
## 3. Evaluation
You can use the following command to evaluate the trained model `export CKPT=/path/to/your/checkpoint.pth`, for example, you can download the our provided checkpoint from [huggingface](https://huggingface.co/hustvl/DiffusionDrive), and set `CKPT=/path/to/downloaded/huggingface_diffusiondrive_agent_ckpt/diffusiondrive_navsim_88p1_PDMS.pth`:
```bash
HYDRA_FULL_ERROR=1 python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=diffusiondrive_agent worker=ray_distributed 'agent.checkpoint_path=/home/users/zhiyu.zheng/workplace/e2ead/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS' experiment_name=diffusiondrive_agent_eval
```

python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=diffusiondrive_agent worker=ray_distributed 'agent.checkpoint_path=/home/users/zhiyu.zheng/workplace/e2ead/DiffusionDrive/epoch\=99-step\=16700.ckpt' experiment_name=diffusiondrive_agent_eval

python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=vanilla_diffusiondrive_agent worker=ray_distributed 'agent.checkpoint_path=/home/users/zhiyu.zheng/workplace/e2ead/DiffusionDrive/epoch\=99-step\=16700.ckpt' experiment_name=vanilla_diffusiondrive_agent_eval