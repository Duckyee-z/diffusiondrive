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


from navsim.visualization.plots import plot_bev_with_traj
import os
import matplotlib.pyplot as plt
from navsim.planning.script.builders.worker_pool_builder import build_worker
# from nuplan.planning.utils.multithreading.worker_utils import worker_map
from typing import Any, Dict, List, Union, Tuple
import uuid

# def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
#     """
#     Helper function to run PDMS evaluation in.
#     :param args: input arguments
#     """
#     save_dir = Path("/home/users/zhiyu.zheng/workplace/e2ead/vdd/diffusiondrive/visualization")
#     node_id = int(os.environ.get("NODE_RANK", 0))
#     thread_id = str(uuid.uuid4())
#     logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

#     log_name = 
#     tokens = [t for a in args for t in a["tokens"]]
#     cfg: DictConfig = args[0]["cfg"]
#     train_data = args[0]['train_data']

#     agent: AbstractAgent = instantiate(cfg.agent)

#     log_save_dir = os.path.join(save_dir, log_name)


#     # tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
#     # pdm_results: List[Dict[str, Any]] = []
#     for idx, token in enumerate(tokens):
#         logger.info(
#             f"Processing scenario {idx + 1} / {len(tokens)} in thread_id={thread_id}, node_id={node_id}"
#         )
#         scene, features, targets = train_data._get_scene_with_token(token)
#         status_feature = features["status_feature"]
#         gt_trajs = targets.get('trajectory') # torch.Size([1, 8, 3])

#         gt_trajs_x = gt_trajs[..., 0] # torch.Size([1, 8])

#         gt_trajs_x_bool = gt_trajs_x[gt_trajs_x>0]

#         if gt_trajs_x_bool.sum() < 8:
#             sum+=1
#         make_sure_dir_exists(Path(log_save_dir))
#         save_path = os.path.join(log_save_dir, f"{idx}_{token}.png")
#         fig, ax = plot_bev_with_traj(scene, gt_trajs[:,0:2].numpy())
#         fig.savefig(save_path)
#         plt.close(fig)



#         pdm_results.append(score_row)
#     return pdm_results




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

    worker = build_worker(cfg)

    logger.info("Building SceneLoader")
    train_scene_loader, train_data = build_datasets(cfg, agent)

    save_dir = Path("/home/users/zhiyu.zheng/workplace/e2ead/vdd/diffusiondrive/visualization")
    # from tqdm import tqdm

    # data_points = [
    #     {
    #         "cfg": cfg,
    #         "log_file": log_file,
    #         "tokens": tokens_list,
    #         "train_data":train_data,
    #     }
    #     for log_file, tokens_list in train_scene_loader.get_tokens_list_per_log().items()
    # ]

    # worker_map(worker, run_pdm_score, data_points)
    from tqdm import tqdm
    sum = 0
    for log_name, tokens in tqdm(train_scene_loader.get_tokens_list_per_log().items()):
        print("=================== Log name:{}, tokens:{} ==========================".format(log_name, len(tokens)))
        log_save_dir = os.path.join(save_dir, log_name)
        
        for idx, token in enumerate(tokens):
            # print("Token: ", token)
            scene, features, targets = train_data._get_scene_with_token(token)
            status_feature = features["status_feature"]
            gt_trajs = targets.get('trajectory') # torch.Size([1, 8, 3])

            gt_trajs_x = gt_trajs[..., 0] # torch.Size([1, 8])

            gt_trajs_x_bool = gt_trajs_x[gt_trajs_x>0]

            if gt_trajs_x_bool.sum() < 8:
                sum+=1
                make_sure_dir_exists(Path(log_save_dir))
                save_path = os.path.join(log_save_dir, f"{idx}_{token}.png")
                fig, ax = plot_bev_with_traj(scene, gt_trajs[:,0:2].numpy())
                fig.savefig(save_path)
                plt.close(fig)




            # # status_feature [driving_command, ego_velocity, ego_acceleration] = [_, _, _, _, vx, vy, ax, ay]
            # velo_now = status_feature[..., 4:6] # torch.Size([64, 2])
            # velo_now_x = velo_now[0]
            # mean_anchor_x = velo_now_x * torch.arange(1, 9) / 2
            # mean_anchor = torch.stack([
            #     mean_anchor_x,
            #     torch.zeros_like(mean_anchor_x),
            # ], dim = -1)

            # # print(velo_now.shape)

            # gt_trajs = targets.get('trajectory') # torch.Size([64, 8, 3])
            # velocity = targets.get('velocity') # torch.Size([64, 8, 2])
            # acceleration = targets.get('acceleration') # torch.Size([64, 8, 2])
            # velo_now = torch.tensor([velo_now_x.item(), 0.0])
            
            # velo_now_x = velo_now[0].unsqueeze(0) # torch.Size([1])
            # velo_now = torch.cat([velo_now_x, torch.zeros_like(velo_now_x)], dim=0).unsqueeze(0)
            # acceleration = acceleration.unsqueeze(0) # torch.Size([1, 8, 2])


            # fig, ax = plot_bev_with_pred_traj(scene, mean_anchor.numpy(), traj2.squeeze(0).numpy())

            
    print(sum)

if __name__ == "__main__":
    main()
