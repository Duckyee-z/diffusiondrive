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
import torch

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[SceneLoader, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        if cfg.debug:
            train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names[:8] if log_name in cfg.train_logs
            ]
        else:
            train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
            ]

    else:
        train_scene_filter.log_names = cfg.train_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_scene_loader, train_data

def make_sure_dir_exists(path: Path) -> None:
    """
    Make sure the directory exists.
    :param path: path to the directory
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    else:
        logger.info(f"Directory already exists: {path}")


from navsim.visualization.plots import plot_bev_with_pred_traj
import os
import matplotlib.pyplot as plt

def high_order_integration(accelerations, initial_velocity=None, dt=0.5):
    """
    Integrate accelerations to get velocities
    
    Args:
        accelerations: Tensor of shape [B, T, 2] containing ax, ay acceleration components
        initial_velocity: Optional initial velocities of shape [B, 2]. If None, assumes zeros
        dt: Time step between samples
        
    Returns:
        Tensor of shape [B, T, 2] containing vx, vy velocity components
    """
    batch_size, time_steps, dims = accelerations.shape
    device = accelerations.device
    
    # 设定初始速度
    if initial_velocity is None:
        initial_velocity = torch.zeros(batch_size, dims, device=device)
    else:
        # 确保维度正确
        if initial_velocity.dim() == 2:
            initial_velocity = initial_velocity  # [B, 2]
        else:
            initial_velocity = initial_velocity.squeeze(1)  # 防止 [B, 1, 2] 的情况
    
    # 转换为 [B, 1, 2] 用于拼接
    initial_velocity = initial_velocity.unsqueeze(1)
    
    # 使用梯形法则积分
    avg_accelerations = 0.5 * (accelerations[:, :-1] + accelerations[:, 1:])  # [B, T-1, 2]
    
    # 计算每一步的速度变化
    velocity_changes = avg_accelerations * dt  # [B, T-1, 2]
    
    # 计算速度的累积和
    cumulative_velocity = torch.cumsum(velocity_changes, dim=1)  # [B, T-1, 2]
    
    # 构建完整速度序列，从初始速度开始
    velocities = torch.cat([initial_velocity, initial_velocity + cumulative_velocity], dim=1)  # [B, T, 2]
    
    return velocities

def high_order_to_trajectory(velocities, initial_position=None, initial_velocity=None, dt=0.5, order=1):
    """
    Convert velocity sequence to position trajectory through integration.
    
    Args:
        velocities: Tensor of shape [B, T, 2] containing vx, vy velocity components
        initial_position: Optional initial positions of shape [B, 2]. If None, assumes zeros
        initial_velocity: Optional initial velocities of shape [B, 2]. If None, assumes zeros
        dt: Time step between samples (default: 0.1s)
        order: Integration order (1: velocity to position, 2: acceleration to position)
        
    Returns:
        Tensor of shape [B, T, 2] containing x, y position coordinates
    """
    batch_size, time_steps, dims = velocities.shape
    device = velocities.device
    
    # 高阶积分处理 (从加速度到速度)
    if order == 2:
        # 首先将加速度积分为速度
        velocities_from_accel = high_order_integration(velocities, initial_velocity, dt)
        # 然后将速度积分为位置
        return high_order_to_trajectory(velocities_from_accel, initial_position, None, dt, order=1)
    
    # 处理初始位置
    if initial_position is None:
        initial_position = torch.zeros(batch_size, dims, device=device)
    else:
        # 确保初始位置维度正确
        if initial_position.dim() == 2:
            initial_position = initial_position  # [B, 2]
        else:
            initial_position = initial_position.squeeze(1)  # 防止 [B, 1, 2] 的情况
    
    # 使用梯形法则积分
    # 计算相邻时间点的平均速度
    avg_velocities = 0.5 * (velocities[:, :-1] + velocities[:, 1:])  # [B, T-1, 2]
    
    # 计算每一步的位移 (平均速度 * dt)
    displacements = avg_velocities * dt  # [B, T-1, 2]
    
    # 计算位移的累积和
    cumulative_displacements = torch.cumsum(displacements, dim=1)  # [B, T-1, 2]
    
    # 将初始位置添加到第一个位置
    initial_position = initial_position.unsqueeze(1)  # [B, 1, 2]
    
    # 构建完整轨迹：初始位置加上累积位移
    trajectories = torch.cat([initial_position, initial_position + cumulative_displacements], dim=1)  # [B, T, 2]
    
    return trajectories


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

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    logger.info("Building SceneLoader")
    train_scene_loader, train_data = build_datasets(cfg, agent)

    save_dir = Path("/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/visualization")

    for log_name, tokens in train_scene_loader.get_tokens_list_per_log().items():
        print("=================== Log name:{}, tokens:{} ==========================".format(log_name, len(tokens)))
        log_save_dir = os.path.join(save_dir, log_name)
        make_sure_dir_exists(Path(log_save_dir))
        for idx, token in enumerate(tokens):
            print("Token: ", token)
            scene, features, targets = train_data._get_scene_with_token(token)
            status_feature = features["status_feature"]

            # status_feature [driving_command, ego_velocity, ego_acceleration] = [_, _, _, _, vx, vy, ax, ay]
            velo_now = status_feature[..., 4:6] # torch.Size([64, 2])
            velo_now_x = velo_now[0]
            mean_anchor_x = velo_now_x * torch.arange(1, 9) / 2
            mean_anchor = torch.stack([
                mean_anchor_x,
                torch.zeros_like(mean_anchor_x),
            ], dim = -1)

            # print(velo_now.shape)
            acc_now = status_feature[..., 6:8] # torch.Size([64, 2])

            gt_trajs = targets.get('trajectory') # torch.Size([64, 8, 3])
            velocity = targets.get('velocity') # torch.Size([64, 8, 2])
            acceleration = targets.get('acceleration') # torch.Size([64, 8, 2])
            velo_now = torch.tensor([velo_now_x.item(), 0.0])
            
            velo_now_x = velo_now[0].unsqueeze(0) # torch.Size([1])
            velo_now = torch.cat([velo_now_x, torch.zeros_like(velo_now_x)], dim=0).unsqueeze(0)
            acceleration = acceleration.unsqueeze(0) # torch.Size([1, 8, 2])
            traj2 = high_order_to_trajectory(acceleration, initial_velocity=velo_now, order=2)


            fig, ax = plot_bev_with_pred_traj(scene, mean_anchor.numpy(), traj2.squeeze(0).numpy())

            save_path = os.path.join(log_save_dir, f"{idx}_{token}.png")
            fig.savefig(save_path)
            plt.close(fig)



if __name__ == "__main__":
    main()
