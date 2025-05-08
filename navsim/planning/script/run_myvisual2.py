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


from navsim.visualization.plots import plot_bev_with_agent_multi_outputs
import os
import matplotlib.pyplot as plt

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

    logger.info("Building SceneLoader")
    train_scene_loader, train_data = build_datasets(cfg, agent)

    save_dir = Path("/home/users/zhiyu.zheng/workplace/e2ead/navsim_workplace/exp/visualization_gm")

    for log_name, tokens in train_scene_loader.get_tokens_list_per_log().items():
        print("=================== Log name:{}, tokens:{} ==========================".format(log_name, len(tokens)))
        log_save_dir = os.path.join(save_dir, log_name)
        make_sure_dir_exists(Path(log_save_dir))

        for idx, token in enumerate(tokens):
            print("Token: ", token)
            scene, features, targets = train_data._get_scene_with_token(token)

            fig, ax = plot_bev_with_agent_multi_outputs(scene, agent)

            save_path = os.path.join(log_save_dir, f"{idx}_{token}.png")
            fig.savefig(save_path)
            plt.close(fig)



if __name__ == "__main__":
    main()
