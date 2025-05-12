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


    x_only_velo_traj_list = []
    speed_velo_traj_list = []
    ego_velo_traj_list = []
    trajs_list = []

    for it, batch in enumerate(tqdm(train_dataloader)):
        # print(it)
        features, targets = batch
        now_ego_velo = features["status_feature"][..., 4:6]

        dt = 0.5
        num_poses = targets['trajectory'].shape[1]

        trajs = targets['trajectory'][...,:2].numpy() # torch.Size([64, 8, 2])

        velo_now_x = now_ego_velo[..., 0]

        x_only_velo = torch.stack([velo_now_x, torch.zeros_like(velo_now_x)], dim=-1).numpy()

        
        ego_velo = now_ego_velo.numpy()

        t = np.linspace(0, 4, 9)
        x_only_traj = (x_only_velo[:, np.newaxis, :] * t[np.newaxis, :, np.newaxis])[:,1:,:]
        ego_velo_traj = (ego_velo[:, np.newaxis, :] * t[np.newaxis, :, np.newaxis])[:,1:,:]
        # x_only_velo_traj = np.array(
        #     [[(time_idx + 1) * dt * x_only_velo] for time_idx in range(num_poses)],
        #     dtype=np.float32,
        # )
        # speed_velo_traj = np.array(
        #     [[(time_idx + 1) * dt * speed_velo] for time_idx in range(num_poses)],
        #     dtype=np.float32,
        # )

        # ego_velo_traj = np.array(
        #     [[(time_idx + 1) * dt * ego_velo] for time_idx in range(num_poses)],
        #     dtype=np.float32,
        # )
        x_only_velo_traj_list.append(x_only_traj)

        ego_velo_traj_list.append(ego_velo_traj)
        trajs_list.append(trajs)


    # train_acc_np, train_velo_np = np.concatenate(train_acc, axis=0), np.concatenate(train_velo, axis=0)
    # print(train_acc_np.shape, train_velo_np.shape)
    x_only_velo_traj_list = np.concatenate(x_only_velo_traj_list, axis=0)
    ego_velo_traj_list = np.concatenate(ego_velo_traj_list, axis=0)
    trajs_list = np.concatenate(trajs_list, axis=0)

    print(x_only_velo_traj_list.shape, ego_velo_traj_list.shape, trajs_list.shape)

    train_dict = dict(
        x_only_velo_traj_list=x_only_velo_traj_list,
        speed_velo_traj_list=speed_velo_traj_list,
        ego_velo_traj_list=ego_velo_traj_list,
        trajs_list=trajs_list
    )
    import pickle
    with open("./tb_logs/speed_anchor.pkl", "wb") as f:
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
