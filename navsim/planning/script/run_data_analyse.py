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


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


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

    # logger.info("Building Lightning Module")
    # lightning_module = AgentLightningModule(
    #     agent=agent,
    # )

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
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            debug=cfg.debug
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))

    # logger.info("Building Trainer")
    # trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())


    train_velocity_x, train_velocity_y = [], []
    train_acceleration_x, train_acceleration_y = [], []
    train_trajectory_x, train_trajectory_y = [], []
    train_trajectory_0_x, train_trajectory_0_y = [], []
    train_trajectory_1_x, train_trajectory_1_y = [], []
    train_trajectory_2_x, train_trajectory_2_y = [], []
    train_trajectory_3_x, train_trajectory_3_y = [], []
    train_trajectory_4_x, train_trajectory_4_y = [], []
    train_trajectory_5_x, train_trajectory_5_y = [], []
    train_trajectory_6_x, train_trajectory_6_y = [], []
    train_trajectory_7_x, train_trajectory_7_y = [], []

    for it, batch in enumerate(tqdm(train_dataloader)):
        # print(it)
        features, targets = batch
        status_feature: torch.Tensor = features["status_feature"] # B,8
        trajectory: torch.Tensor = targets["trajectory"] # 
    
        velocity = status_feature[..., 4:6] # B,3
        acceleration = status_feature[..., 6:8] # B,3

        velocity_x, velocity_y = velocity[..., 0].numpy(), velocity[..., 1].numpy()
        acceleration_x, acceleration_y = acceleration[..., 0].numpy(), acceleration[..., 1].numpy()
        trajectory_x, trajectory_y = trajectory[..., 0].numpy(), trajectory[..., 1].numpy()
        trajectory_0_x, trajectory_0_y = trajectory[:, 0, 0].numpy(), trajectory[:, 0, 1].numpy()
        trajectory_1_x, trajectory_1_y = trajectory[:, 1, 0].numpy(), trajectory[:, 1, 1].numpy()
        trajectory_2_x, trajectory_2_y = trajectory[:, 2, 0].numpy(), trajectory[:, 2, 1].numpy()
        trajectory_3_x, trajectory_3_y = trajectory[:, 3, 0].numpy(), trajectory[:, 3, 1].numpy()
        trajectory_4_x, trajectory_4_y = trajectory[:, 4, 0].numpy(), trajectory[:, 4, 1].numpy()
        trajectory_5_x, trajectory_5_y = trajectory[:, 5, 0].numpy(), trajectory[:, 5, 1].numpy()
        trajectory_6_x, trajectory_6_y = trajectory[:, 6, 0].numpy(), trajectory[:, 6, 1].numpy()
        trajectory_7_x, trajectory_7_y = trajectory[:, 7, 0].numpy(), trajectory[:, 7, 1].numpy()
        train_velocity_x.append(velocity_x), train_velocity_y.append(velocity_y)
        train_acceleration_x.append(acceleration_x), train_acceleration_y.append(acceleration_y)
        train_trajectory_x.append(trajectory_x), train_trajectory_y.append(trajectory_y)
        train_trajectory_0_x.append(trajectory_0_x), train_trajectory_0_y.append(trajectory_0_y)
        train_trajectory_1_x.append(trajectory_1_x), train_trajectory_1_y.append(trajectory_1_y)
        train_trajectory_2_x.append(trajectory_2_x), train_trajectory_2_y.append(trajectory_2_y)
        train_trajectory_3_x.append(trajectory_3_x), train_trajectory_3_y.append(trajectory_3_y)
        train_trajectory_4_x.append(trajectory_4_x), train_trajectory_4_y.append(trajectory_4_y)
        train_trajectory_5_x.append(trajectory_5_x), train_trajectory_5_y.append(trajectory_5_y)
        train_trajectory_6_x.append(trajectory_6_x), train_trajectory_6_y.append(trajectory_6_y)
        train_trajectory_7_x.append(trajectory_7_x), train_trajectory_7_y.append(trajectory_7_y)

    train_velocity_x, train_velocity_y = np.concatenate(train_velocity_x, axis=0), np.concatenate(train_velocity_y, axis=0)
    train_acceleration_x, train_acceleration_y = np.concatenate(train_acceleration_x, axis=0), np.concatenate(train_acceleration_y, axis=0)
    train_trajectory_x, train_trajectory_y = np.concatenate(train_trajectory_x, axis=0), np.concatenate(train_trajectory_y, axis=0)
    train_trajectory_0_x, train_trajectory_0_y = np.concatenate(train_trajectory_0_x, axis=0), np.concatenate(train_trajectory_0_y, axis=0)
    train_trajectory_1_x, train_trajectory_1_y = np.concatenate(train_trajectory_1_x, axis=0), np.concatenate(train_trajectory_1_y, axis=0)
    train_trajectory_2_x, train_trajectory_2_y = np.concatenate(train_trajectory_2_x, axis=0), np.concatenate(train_trajectory_2_y, axis=0)
    train_trajectory_3_x, train_trajectory_3_y = np.concatenate(train_trajectory_3_x, axis=0), np.concatenate(train_trajectory_3_y, axis=0)
    train_trajectory_4_x, train_trajectory_4_y = np.concatenate(train_trajectory_4_x, axis=0), np.concatenate(train_trajectory_4_y, axis=0)
    train_trajectory_5_x, train_trajectory_5_y = np.concatenate(train_trajectory_5_x, axis=0), np.concatenate(train_trajectory_5_y, axis=0)
    train_trajectory_6_x, train_trajectory_6_y = np.concatenate(train_trajectory_6_x, axis=0), np.concatenate(train_trajectory_6_y, axis=0)
    train_trajectory_7_x, train_trajectory_7_y = np.concatenate(train_trajectory_7_x, axis=0), np.concatenate(train_trajectory_7_y, axis=0)

    train_dict = dict(
        train_velocity_x=train_velocity_x,
        train_velocity_y=train_velocity_y,
        train_acceleration_x=train_acceleration_x,
        train_acceleration_y=train_acceleration_y,
        train_trajectory_x=train_trajectory_x,
        train_trajectory_y=train_trajectory_y,
        train_trajectory_0_x=train_trajectory_0_x,
        train_trajectory_0_y=train_trajectory_0_y,
        train_trajectory_1_x=train_trajectory_1_x,
        train_trajectory_1_y=train_trajectory_1_y,
        train_trajectory_2_x=train_trajectory_2_x,
        train_trajectory_2_y=train_trajectory_2_y,
        train_trajectory_3_x=train_trajectory_3_x,
        train_trajectory_3_y=train_trajectory_3_y,
        train_trajectory_4_x=train_trajectory_4_x,
        train_trajectory_4_y=train_trajectory_4_y,
        train_trajectory_5_x=train_trajectory_5_x,
        train_trajectory_5_y=train_trajectory_5_y,
        train_trajectory_6_x=train_trajectory_6_x,
        train_trajectory_6_y=train_trajectory_6_y,
        train_trajectory_7_x=train_trajectory_7_x,
        train_trajectory_7_y=train_trajectory_7_y
    )
    import pickle
    with open("train.pkl", "wb") as f:
        pickle.dump(train_dict, f)

    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    val_velocity_x, val_velocity_y = [], []
    val_acceleration_x, val_acceleration_y = [], []
    val_trajectory_x, val_trajectory_y = [], []
    val_trajectory_0_x, val_trajectory_0_y = [], []
    val_trajectory_1_x, val_trajectory_1_y = [], []
    val_trajectory_2_x, val_trajectory_2_y = [], []
    val_trajectory_3_x, val_trajectory_3_y = [], []
    val_trajectory_4_x, val_trajectory_4_y = [], []
    val_trajectory_5_x, val_trajectory_5_y = [], []
    val_trajectory_6_x, val_trajectory_6_y = [], []
    val_trajectory_7_x, val_trajectory_7_y = [], []


    for it, batch in enumerate(tqdm(val_dataloader)):
        # print(it)
        features, targets = batch
        status_feature: torch.Tensor = features["status_feature"] # B,8
        trajectory: torch.Tensor = targets["trajectory"] # 
    
        velocity = status_feature[..., 4:6] # B,3
        acceleration = status_feature[..., 6:8] # B,3

        velocity_x, velocity_y = velocity[..., 0].numpy(), velocity[..., 1].numpy()
        acceleration_x, acceleration_y = acceleration[..., 0].numpy(), acceleration[..., 1].numpy()
        trajectory_x, trajectory_y = trajectory[..., 0].numpy(), trajectory[..., 1].numpy()
        trajectory_0_x, trajectory_0_y = trajectory[:, 0, 0].numpy(), trajectory[:, 0, 1].numpy()
        trajectory_1_x, trajectory_1_y = trajectory[:, 1, 0].numpy(), trajectory[:, 1, 1].numpy()
        trajectory_2_x, trajectory_2_y = trajectory[:, 2, 0].numpy(), trajectory[:, 2, 1].numpy()
        trajectory_3_x, trajectory_3_y = trajectory[:, 3, 0].numpy(), trajectory[:, 3, 1].numpy()
        trajectory_4_x, trajectory_4_y = trajectory[:, 4, 0].numpy(), trajectory[:, 4, 1].numpy()
        trajectory_5_x, trajectory_5_y = trajectory[:, 5, 0].numpy(), trajectory[:, 5, 1].numpy()
        trajectory_6_x, trajectory_6_y = trajectory[:, 6, 0].numpy(), trajectory[:, 6, 1].numpy()
        trajectory_7_x, trajectory_7_y = trajectory[:, 7, 0].numpy(), trajectory[:, 7, 1].numpy()

        val_velocity_x.append(velocity_x), val_velocity_y.append(velocity_y)
        val_acceleration_x.append(acceleration_x), val_acceleration_y.append(acceleration_y)
        val_trajectory_x.append(trajectory_x), val_trajectory_y.append(trajectory_y)
        val_trajectory_0_x.append(trajectory_0_x), val_trajectory_0_y.append(trajectory_0_y)
        val_trajectory_1_x.append(trajectory_1_x), val_trajectory_1_y.append(trajectory_1_y)
        val_trajectory_2_x.append(trajectory_2_x), val_trajectory_2_y.append(trajectory_2_y)
        val_trajectory_3_x.append(trajectory_3_x), val_trajectory_3_y.append(trajectory_3_y)
        val_trajectory_4_x.append(trajectory_4_x), val_trajectory_4_y.append(trajectory_4_y)
        val_trajectory_5_x.append(trajectory_5_x), val_trajectory_5_y.append(trajectory_5_y)
        val_trajectory_6_x.append(trajectory_6_x), val_trajectory_6_y.append(trajectory_6_y)
        val_trajectory_7_x.append(trajectory_7_x), val_trajectory_7_y.append(trajectory_7_y)
    
    val_velocity_x, val_velocity_y = np.concatenate(val_velocity_x, axis=0), np.concatenate(val_velocity_y, axis=0)
    val_acceleration_x, val_acceleration_y = np.concatenate(val_acceleration_x, axis=0), np.concatenate(val_acceleration_y, axis=0)
    val_trajectory_x, val_trajectory_y = np.concatenate(val_trajectory_x, axis=0), np.concatenate(val_trajectory_y, axis=0)
    val_trajectory_0_x, val_trajectory_0_y = np.concatenate(val_trajectory_0_x, axis=0), np.concatenate(val_trajectory_0_y, axis=0)
    val_trajectory_1_x, val_trajectory_1_y = np.concatenate(val_trajectory_1_x, axis=0), np.concatenate(val_trajectory_1_y, axis=0)
    val_trajectory_2_x, val_trajectory_2_y = np.concatenate(val_trajectory_2_x, axis=0), np.concatenate(val_trajectory_2_y, axis=0)
    val_trajectory_3_x, val_trajectory_3_y = np.concatenate(val_trajectory_3_x, axis=0), np.concatenate(val_trajectory_3_y, axis=0)
    val_trajectory_4_x, val_trajectory_4_y = np.concatenate(val_trajectory_4_x, axis=0), np.concatenate(val_trajectory_4_y, axis=0)
    val_trajectory_5_x, val_trajectory_5_y = np.concatenate(val_trajectory_5_x, axis=0), np.concatenate(val_trajectory_5_y, axis=0)
    val_trajectory_6_x, val_trajectory_6_y = np.concatenate(val_trajectory_6_x, axis=0), np.concatenate(val_trajectory_6_y, axis=0)
    val_trajectory_7_x, val_trajectory_7_y = np.concatenate(val_trajectory_7_x, axis=0), np.concatenate(val_trajectory_7_y, axis=0)

    val_dict = dict(
        val_velocity_x=val_velocity_x,
        val_velocity_y=val_velocity_y,
        val_acceleration_x=val_acceleration_x,
        val_acceleration_y=val_acceleration_y,
        val_trajectory_x=val_trajectory_x,
        val_trajectory_y=val_trajectory_y,
        val_trajectory_0_x=val_trajectory_0_x,
        val_trajectory_0_y=val_trajectory_0_y,
        val_trajectory_1_x=val_trajectory_1_x,
        val_trajectory_1_y=val_trajectory_1_y,
        val_trajectory_2_x=val_trajectory_2_x,
        val_trajectory_2_y=val_trajectory_2_y,
        val_trajectory_3_x=val_trajectory_3_x,
        val_trajectory_3_y=val_trajectory_3_y,
        val_trajectory_4_x=val_trajectory_4_x,
        val_trajectory_4_y=val_trajectory_4_y,
        val_trajectory_5_x=val_trajectory_5_x,
        val_trajectory_5_y=val_trajectory_5_y,
        val_trajectory_6_x=val_trajectory_6_x,
        val_trajectory_6_y=val_trajectory_6_y,
        val_trajectory_7_x=val_trajectory_7_x,
        val_trajectory_7_y=val_trajectory_7_y
    )
    with open("val.pkl", "wb") as f:
        pickle.dump(val_dict, f)


    # logger.info("Starting Training")
    # trainer.fit(
    #     model=lightning_module,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    # )


if __name__ == "__main__":
    main()
