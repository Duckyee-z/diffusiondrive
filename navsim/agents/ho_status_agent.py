from typing import Any, List, Dict, Optional, Union, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.vddrive_ho.vddrive_utils import high_order_to_trajectory


from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Scene, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
import numpy as np

class HOAgentFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        status_feature = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )
        return {"ego_status": status_feature}

class HOAgentTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        velocity, acceleration = self.get_velo_accel(scene, frame_idx, self._trajectory_sampling.num_poses)


        return {"trajectory": torch.tensor(future_trajectory.poses),
                "velocity": velocity, 
                "acceleration": acceleration
                }

    def get_velo_accel(self, scene: Scene, start_frame_idx, num_frames) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inherited, see superclass."""
        # velocity_list = []
        # acceleration_list = []
        velocity_list = []
        acceleration_list = []
        for frame_idx in range(start_frame_idx + 1, start_frame_idx + num_frames + 1):
            velocity_list.append(scene.frames[frame_idx].ego_status.ego_velocity)
            acceleration_list.append(scene.frames[frame_idx].ego_status.ego_acceleration) 
        velocity = torch.tensor(np.array(velocity_list), dtype=torch.float32)
        acceleration = torch.tensor(np.array(acceleration_list), dtype=torch.float32)

        return (velocity, acceleration)


class HOAgent(AbstractAgent):
    """EgoStatMLP agent interface."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        # hidden_layer_dim: int,
        # lr: float,
        # checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param trajectory_sampling: trajectory sampling specification.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.requires_scene = True

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [HOAgentTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [HOAgentFeatureBuilder()]
    
    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()
    
    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_target_builders():
            targets.update(builder.compute_targets(scene))

        # add batch dimension
        features = {k: v for k, v in features.items()}
        targets = {k: v for k, v in targets.items()}

        status_feature: torch.Tensor = features["ego_status"]
        velo_ego = status_feature[..., 4:6]
        acc_ego = status_feature[..., 6:8]
        
         # torch.Size([64, 2])
        velo_now_x = velo_ego[..., 0]
        velo_now = torch.stack([velo_now_x, torch.zeros_like(velo_now_x)], dim=-1)

        gt_poses = targets['trajectory']
        velocity = targets.get('velocity') # torch.Size([64, 8, 2])
        acceleration = targets.get('acceleration') # torch.Size([64, 8, 2])

        # print(acceleration.numpy().shape, velocity.numpy().shape, acc_ego.numpy().shape, velo_ego.numpy().shape)
        pred_traj = acc_to_trajectory(a=acceleration.numpy(), v=velocity.numpy(), a0=acc_ego.numpy(), v0=velo_now.numpy())

        # pred_traj = vel_to_trajectory(v=velocity.numpy(), v0=velo_now.numpy())

        # print(gt_poses[..., 2:].numpy().shape)
        pred_traj = np.concatenate([pred_traj, gt_poses[..., 2:].numpy()], axis=-1)
        # pred_traj = high_order_to_trajectory(velocities = acceleration.unsqueeze(1), initial_position=None, initial_velocity=velo_ego, dt=0.5, order=2)
        # print(pred_traj.shape,)
        # pred_traj = torch.cat([pred_traj.squeeze(1), gt_poses[..., 2].unsqueeze(-1)], dim=-1)


        # extract trajectory
        return Trajectory(pred_traj)


def acc_to_trajectory(a, v, a0, v0, dt=0.5):

    v = np.zeros((9, 2))  # 包括初始速度v0和8个时间步
    s = np.zeros((9, 2))  # 包括初始位移s0和8个时间步


    v[0] = v0
    s[0] = np.array([0, 0])  # 假设初始位移为0

    # 计算速度
    for i in range(8):
        if i == 0:
            a_prev = a0.squeeze()
        else:
            a_prev = a[i-1]
        v[i+1] = v[i] + (a_prev + a[i]) / 2 * dt  # 梯形法则积分

    vx, vy = v[:, 0], v[:, 1]
    vx[vx<0] = 0
    v = np.stack([vx, vy], axis=-1)

    # 计算位移
    for i in range(8):
        s[i+1] = s[i] + (v[i] + v[i+1]) / 2 * dt  # 梯形法则积分

    # 提取结果（去掉初始值）
    velocity = v[1:]  # shape=(8, 2)
    displacement = s[1:]  # shape=(8, 2)

    return displacement

def vel_to_trajectory(v, v0, dt=0.5):
    
    # 初始化位移数组（包括初始状态）
    s = np.zeros((9, 2))  # s[0] = [0, 0], s[1:] 为计算结果
    
    # 计算位移（梯形法则积分速度）
    for i in range(8):
        if i == 0:
            v_prev = v0  # 使用初始速度 v0
        else:
            v_prev = v[i-1]
        s[i+1] = s[i] + 0.5 * (v_prev + v[i]) * dt
    
    return s[1:]  # 返回8个时间步的位移（去掉初始状态）
