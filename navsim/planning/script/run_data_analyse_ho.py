from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

from tqdm import tqdm
import torch
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


anchor_minmax = torch.tensor([
    # Point 0
    [[-1.53, 3.92],   # x: min=-1.53, max=3.92
     [-0.40, 0.98]],  # y: min=-0.40, max=0.98
    # Point 1
    [[-1.94, 4.79],
     [-1.62, 2.16]],
    # Point 2
    [[-4.51, 9.96],
     [-3.43, 3.96]],
    # Point 3
    [[-8.06, 10.24],
     [-5.85, 6.59]],
    # Point 4
    [[-12.18, 10.73],
     [-8.59, 9.93]],
    # Point 5
    [[-16.62, 14.17],
     [-11.95, 13.63]],
    # Point 6
    [[-22.47, 18.02],
     [-15.71, 17.83]],
    # Point 7
    [[-28.36, 22.24],
     [-19.68, 22.32]]
], dtype=torch.float32)



def norm_odo(odo_info_fut, vec=anchor_minmax): # odo_info_fut ([64, 20, 8, 2])
        """
        对输入张量的每个点的x和y坐标进行Min-Max归一化。
        args:
            tensor (torch.Tensor): 形状为(B, L, 8, 2)的输入张量。
            vec (torch.Tensor): 形状为(8, 2, 2)的极值张量, 其中vec[i, j, 0]为最小值, vec[i, j, 1]为最大值。
        
        return:
            torch.Tensor: 归一化后的张量，形状与输入相同。
        """
        # 提取每个点的x和y的min、max
        x_mins = vec[:, 0, 0].to(odo_info_fut.device)  # 形状 (8,)
        x_maxs = vec[:, 0, 1].to(odo_info_fut.device)
        y_mins = vec[:, 1, 0].to(odo_info_fut.device)
        y_maxs = vec[:, 1, 1].to(odo_info_fut.device)

        # 分离x和y坐标
        x_coords = odo_info_fut[..., 0]  # 形状 (B, L, 8)
        y_coords = odo_info_fut[..., 1]

        # 计算归一化后的坐标，利用广播机制
        x_range = x_maxs - x_mins
        normalized_x = 10* (x_coords - x_mins[None, None, :]) / x_range[None, None, :] - 5

        y_range = y_maxs - y_mins
        normalized_y = 10* (y_coords - y_mins[None, None, :]) / y_range[None, None, :] - 5

        # 合并结果并保持原有维度

        return torch.stack([normalized_x, normalized_y], dim=-1)

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"

        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            debug=cfg.debug
        )
        # val_data = CacheOnlyDataset(
        #     cache_path=cfg.cache_path,
        #     feature_builders=agent.get_feature_builders(),
        #     target_builders=agent.get_target_builders(),
        #     log_names=cfg.val_logs,
        #     debug=cfg.debug
        # )
    logger.info("Building Datasets")

    # val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    # logger.info("Num validation samples: %d", len(val_data))

    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))


    odo_bias_list = []
    gt_list = []
    speed_anchor_list = []
    velo_list = []
    driving_command_list = []

    for it, batch in enumerate(tqdm(train_dataloader)):
        # print(it)
        features, targets = batch
        n_trajs = 1

        gt_trajs = targets.get('trajectory').float() 
        driving_command =  features["status_feature"][..., :4].numpy()
        velo_ego = features["status_feature"][..., 4:5]
        velo = torch.stack([velo_ego, torch.zeros_like(velo_ego)], dim=-1)
        speed_anchor = velo.unsqueeze(1) * torch.linspace(0.5, 4, steps=8).unsqueeze(0).unsqueeze(-1)
        speed_anchor = speed_anchor.squeeze(1)

        plan_anchor = torch.stack([gt_trajs[...,:2]-speed_anchor]*n_trajs, dim=1).float() # torch.Size([64, 20, 8, 2])

        # odo_bias = norm_odo(plan_anchor, vec=anchor_minmax)
        odo_bias = (gt_trajs[...,:2]-speed_anchor).numpy()

        gt_list.append(gt_trajs[...,:2].numpy())
        speed_anchor_list.append(speed_anchor.numpy())
        velo_list.append(velo.numpy())
        odo_bias_list.append(odo_bias)
        driving_command_list.append(driving_command)



    # train_acc_np, train_velo_np = np.concatenate(train_acc, axis=0), np.concatenate(train_velo, axis=0)
    # print(train_acc_np.shape, train_velo_np.shape)
    # odo_bias_list = np.concatenate(odo_bias_list, axis=0)
    gt_list = np.concatenate(gt_list, axis=0)
    velo_list = np.concatenate(velo_list, axis=0)
    speed_anchor_list = np.concatenate(speed_anchor_list, axis=0)
    odo_bias_list = np.concatenate(odo_bias_list, axis=0)
    driving_command_list = np.concatenate(driving_command_list, axis=0)

    train_dict = dict(
        gt_trajs=gt_list, 
        velo_list=velo_list,
        speed_anchor_list=speed_anchor_list,
        odo_bias_list=odo_bias_list,
        driving_command_list = driving_command_list
    )
    import pickle
    with open("./tb_logs/speed_anchor_data.pkl", "wb") as f:
        pickle.dump(train_dict, f)


    # val_acc = []
    # val_velo = []
    
    # for it, batch in enumerate(tqdm(val_dataloader)):
    #     # print(it)
    #     features, targets = batch

    #     velocity = targets.get('velocity').numpy() # torch.Size([64, 8, 2]) 
    #     acceleration = targets.get('acceleration').numpy() # torch.Size([64, 8, 2])

    #     val_acc.append(acceleration)
    #     val_velo.append(velocity)

    # val_acc_np, val_velo_np = np.concatenate(val_acc, axis=0), np.concatenate(val_velo, axis=0)
    # print(val_acc_np.shape, val_velo_np.shape)


    # val_dict = dict(
    #     val_acc=val_acc_np,
    #     val_velo=val_velo_np
    # )
    # with open("./tb_logs/val_ho.pkl", "wb") as f:
    #     pickle.dump(val_dict, f)


    # logger.info("Starting Training")
    # trainer.fit(
    #     model=lightning_module,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    # )


if __name__ == "__main__":
    main()
