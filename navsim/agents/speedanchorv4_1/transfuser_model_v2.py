from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.speedanchorv4_1.transfuser_config import TransfuserConfig
from navsim.agents.speedanchorv4_1.transfuser_backbone import TransfuserBackbone
from navsim.agents.speedanchorv4_1.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.speedanchorv4_1.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.speedanchorv4_1.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.speedanchorv4_1.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union

import logging
logger = logging.getLogger(__name__)

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


point_minmax = torch.tensor([
    # Point 0
    [[-0.40, 8.58],   # x: min=-0.40, max=8.58
     [-0.40, 0.98]],  # y: min=-0.40, max=0.98

    # Point 1
    [[-0.77, 16.63],
     [-1.62, 2.16]],

    # Point 2
    [[-1.12, 24.85],
     [-3.43, 3.96]],

    # Point 3
    [[-1.39, 33.01],
     [-5.85, 6.59]],

    # Point 4
    [[-1.53, 41.13],
     [-8.59, 9.93]],

    # Point 5
    [[-1.57, 49.10],
     [-11.95, 13.63]],

    # Point 6
    [[-1.55, 57.15],
     [-15.71, 17.83]],

    # Point 7
    [[-1.38, 65.17],
     [-19.68, 22.32]]
], dtype=torch.float32)




class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)
        logger.info("================= Init VDDrive-v2.3 maxnorm ====================")
        # logger.info("Random init: {}".format(config.random_init))
        logger.info("infer_step_num: {}".format(config.infer_step_num))
        logger.info("Speed anchor: {}".format(config.speed_anchor))
        logger.info("use_diffusion_loss: {}".format(config.use_diffusion_loss))
        logger.info("random_scale: {}".format(config.random_scale))
        logger.info("infer_minus_anchor: {}".format(config.infer_minus_anchor))
        logger.info("anchor_embed: {}".format(config.anchor_embed))
        logger.info("with_query_as_embedding: {}".format(config.with_query_as_embedding))
        logger.info("anchor_embed_interact: {}".format(config.anchor_embed_interact))
        logger.info("use mse loss: {}".format(config.use_mse_loss))
        logger.info("use clamp: {}".format(config.use_clamp))
        logger.info("norm_scale: {}".format(config.norm_scale))
        logger.info("output_result: {}".format(config.output_result))
        logger.info("Truncated vx: {}".format(config.truncated_vx))

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,

            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"] # B,3,256,1024
        lidar_feature: torch.Tensor = features["lidar_feature"] # B,1,256,256
        status_feature: torch.Tensor = features["status_feature"] # B,8
        # [driving_command, ego_velocity, ego_acceleration] = [_, _, _, _, vx, vy, ax, ay]
        # 这里的driving_command是啥 4+2+2

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape, status_encoding[:, None], status_feature, targets=targets, global_img=None)
        output.update(trajectory) # 轨迹预测

        agents = self._agent_head(agents_query) # BEV目标检测
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        # bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)

        traj_delta = self.plan_reg_branch(traj_feature)

        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg
    
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed, # condition
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config : TransfuserConfig,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        # if config.anchor_embed and not config.anchor_embed_interact:
        #     self.time_modulation = ModulationLayer(config.tf_d_model, 256*2)
        # else:
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)

        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg
    
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []

        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg_list.append(poses_reg)
    
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dims=256, drop_ratio=0.1):
        super().__init__()

        num_heads = embed_dims // 32

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
        )
        self.attn_drop = nn.Dropout(drop_ratio)
        self.attn_norm = nn.LayerNorm(embed_dims)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, 2 * embed_dims),
            nn.GELU(),
            nn.Linear(2 * embed_dims, embed_dims),
        )
        self.mlp_drop = nn.Dropout(drop_ratio)
        self.mlp_norm = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, query_pos=None, key_pos=None, key_padding_mask=None, skip=None):
        query = query.permute(1, 0, 2)  # (1, 64, 256)
        key = key.permute(1, 0, 2)  # (20, 64, 256)
        value = value.permute(1, 0, 2)  # (20, 64, 256)
        skip = skip.permute(1, 0, 2)  # (1, 64, 256)
        if query_pos is not None:
            query_pos = query_pos.permute(1, 0, 2)  # (1, 64, 256)
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos


        embed = self.attn_norm(skip + self.attn_drop(self.attn(query, key, value, key_padding_mask=key_padding_mask)[0])).permute(1, 0, 2)
        skip = embed
        embed = self.mlp_norm(skip + self.mlp_drop(self.mlp(embed)))

        return embed


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()
        self.n_trajs = 20
        self.config = config

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
            timestep_spacing=self.config.infer_timestep_spacing
        )

        self.infer_step_num = config.infer_step_num

        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        if self.config.anchor_embed:
            self.anchor_embed = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            if self.config.with_query_as_embedding:
                self.cross_attn = TransformerBlock(embed_dims=d_model, drop_ratio=0.1)
                self.fusion_embed = nn.Sequential(
                    nn.Linear(d_model * 2, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                )
            if self.config.anchor_embed_interact:
                self.anchor_attn = TransformerBlock(embed_dims=d_model, drop_ratio=0.1)


    def norm_odo(self, odo_info_fut, vec=anchor_minmax): # odo_info_fut ([64, 20, 8, 2])
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

        scale=self.config.norm_scale
        scale_mul = scale * 2
        # 分离x和y坐标
        x_coords = odo_info_fut[..., 0]  # 形状 (B, L, 8)
        y_coords = odo_info_fut[..., 1]

        # 计算归一化后的坐标，利用广播机制
        x_range = x_maxs - x_mins
        # before [-1, 1]
        # normalized_x = 2* (x_coords - x_mins[None, None, :]) / x_range[None, None, :] - 1
        # now [-5, 5]
        normalized_x = scale_mul* (x_coords - x_mins[None, None, :]) / x_range[None, None, :] - scale


        y_range = y_maxs - y_mins
        normalized_y = scale_mul* (y_coords - y_mins[None, None, :]) / y_range[None, None, :] - scale

        # 合并结果并保持原有维度

        return torch.stack([normalized_x, normalized_y], dim=-1)

    def denorm_odo(self, odo_info_fut, vec=anchor_minmax):
        # 提取每个点的x和y的min、max
        x_mins = vec[:, 0, 0].to(odo_info_fut.device)  # 形状 (8,)
        x_maxs = vec[:, 0, 1].to(odo_info_fut.device)
        y_mins = vec[:, 1, 0].to(odo_info_fut.device)
        y_maxs = vec[:, 1, 1].to(odo_info_fut.device)

        scale=self.config.norm_scale
        scale_mul = scale * 2
        # 分离x和y坐标
        x_norm = odo_info_fut[..., 0]  # 形状 (B, L, 8)
        y_norm = odo_info_fut[..., 1]

        # 计算归一化后的坐标，利用广播机制
        x_range = x_maxs - x_mins
        denormalized_x = ((x_norm + scale) / scale_mul) * x_range[None, None, :] + x_mins[None, None, :]

        y_range = y_maxs - y_mins
        denormalized_y = ((y_norm + scale) / scale_mul) * y_range[None, None, :] + y_mins[None, None, :]

        return torch.stack([denormalized_x, denormalized_y], dim=-1)
    
    
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, status_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.config.speed_anchor:
            if self.training:
                return self.forward_train_speed_anchor(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,targets,global_img)
            else :
                return self.forward_test_speed_anchor(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,global_img)

        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,targets,global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature, global_img)
    
    def forward_train_speed_anchor(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0] # ego_query torch.Size([64, 1, 256]) agents_query torch.Size([64, 30, 256])
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        gt_trajs = targets.get('trajectory').float() # torch.Size([64, 8, 3])
        n_trajs = self.n_trajs
        velo_ego = status_feature[..., 4:5] # x only
        if self.config.truncated_vx:
            velo_ego[velo_ego < 0] = 0
        velo = torch.stack([velo_ego, torch.zeros_like(velo_ego)], dim=-1).to(device)
        speed_anchor = velo.unsqueeze(1) * torch.linspace(0.5, 4, steps=8).unsqueeze(0).unsqueeze(-1).to(device)
        speed_anchor = speed_anchor.squeeze(1)
        plan_anchor = torch.stack([gt_trajs[...,:2]-speed_anchor]*n_trajs, dim=1).float().to(device) # torch.Size([64, 20, 8, 2])
        odo_info_fut = self.norm_odo(plan_anchor, vec=anchor_minmax) # torch.Size([64, 20, 8, 2])
        odo_info_fut = odo_info_fut.view(bs*self.n_trajs, 8, 2) # torch.Size([1280, 8, 2])
        # import ipdb; ipdb.set_trace()
        timesteps = torch.randint(
            0, 1000,
            (bs * self.n_trajs,), device=device
        ) 
        noise = torch.randn(odo_info_fut.shape, device=device) * self.config.random_scale
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()

        if self.config.use_clamp:
            noisy_traj_points = torch.clamp(noisy_traj_points, min=-1*self.config.norm_scale, max=1*self.config.norm_scale)
        noisy_traj_points = self.denorm_odo(noisy_traj_points, vec=anchor_minmax)
        noisy_traj_points = noisy_traj_points.view(bs, self.n_trajs, 8, 2) # torch.Size([64, 20, 8, 2])
        speed_anchor_stacked=speed_anchor.repeat(1, self.n_trajs, 1, 1).view(bs, self.n_trajs, 8, 2) # torch.Size([64, 20, 8, 2])
        traj_points = torch.concat([noisy_traj_points, speed_anchor_stacked], dim=1) # torch.Size([64, 40, 8, 4])

        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(traj_points,hidden_dim=64) # torch.Size([64, 20, 8, 64])
        traj_pos_embed = traj_pos_embed.flatten(-2) # torch.Size([64, 20, 8, 512])
        traj_feature = self.plan_anchor_encoder(traj_pos_embed) # torch.Size([64, 20, 256])
        traj_feature = traj_feature.view(bs,self.n_trajs*2,-1) # torch.Size([64, 20, 256])
        traj_feature, anchor_embedding = torch.chunk(traj_feature, 2, dim=1) # torch.Size([64, 20, 256]), torch.Size([64, 20, 256])


        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps) # torch.Size([1280, 256])
        time_embed = time_embed.view(bs,self.n_trajs,-1) # torch.Size([64, 20, 256])

        if self.config.anchor_embed:
            time_embed = torch.concat([time_embed, anchor_embedding], dim=-1).to(device) 
            time_embed = self.anchor_embed(time_embed)
            if self.config.with_query_as_embedding:
                ego_query = self.cross_attn(ego_query, agents_query, agents_query, skip=ego_query)
                ego_query = ego_query.repeat(1, self.n_trajs, 1)
                time_embed = self.fusion_embed(torch.cat([ego_query, time_embed], dim=-1))

        # 4. begin the stacked decoder
        poses_reg_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding, global_img)

        # poses_regs = torch.stack(poses_reg_list, dim=0) # torch.Size([64, 20, 8, 3])
        trajectory_loss_dict = {}
        diffusion_loss_dict = {}
        ret_traj_loss, ret_diffusion_loss = 0, 0
        for idx, poses_reg in enumerate(poses_reg_list):
            # trajectory_loss == diffusion loss, i.e, pred x0 in diffusion
            poses = torch.concatenate([
                poses_reg[...,0:2] + speed_anchor.unsqueeze(1), 
                poses_reg[...,2:3]
            ],dim=-1).float()
            if self.config.use_diffusion_loss:
                diffusion_loss = F.mse_loss(poses_reg[...,:2], plan_anchor)
                diffusion_loss_dict[f"diffusion_loss_{idx}"] = diffusion_loss
                ret_diffusion_loss +=diffusion_loss
            if self.config.use_mse_loss:
                trajectory_loss = F.mse_loss(poses, # pred
                                            torch.stack([gt_trajs] * self.n_trajs, dim=1)# gt
                                            )
            else:
                trajectory_loss = F.l1_loss(poses, # pred
                                        torch.stack([gt_trajs] * self.n_trajs, dim=1)# gt
                                        )
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        return {"trajectory": poses, 
                "trajectory_loss":ret_traj_loss,
                "trajectory_loss_dict":trajectory_loss_dict,
                "diffusion_loss":ret_diffusion_loss,
                "diffusion_loss_dict":diffusion_loss_dict}   

    def forward_test_speed_anchor(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, status_feature, global_img) -> Dict[str, torch.Tensor]:

        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(self.infer_step_num, device)
        denoise_steps = list(range(0, self.infer_step_num))

        velo_ego = status_feature[..., 4:5] # x only
        if self.config.truncated_vx:
            velo_ego[velo_ego < 0] = 0
        velo = torch.stack([velo_ego, torch.zeros_like(velo_ego)], dim=-1).to(device)
        speed_anchor = velo.unsqueeze(1) * torch.linspace(0.5, 4, steps=8).unsqueeze(0).unsqueeze(-1).to(device).float()
        # 1. add noise to the plan anchor
        speed_anchor_stacked=speed_anchor


        plan_anchor = torch.randn((bs, 1, 8, 2)).to(device) * self.config.random_scale
        if self.config.infer_minus_anchor:
            plan_anchor = plan_anchor - speed_anchor
        
        img = self.norm_odo(plan_anchor.to(device), vec=anchor_minmax)
        # noise = torch.randn(img.shape, device=device)
        # noisy_trajs = self.denorm_odo(img) 
        ego_fut_mode = img.shape[1]
        traj_list = {}
        for step_idx in denoise_steps:
            k = self.diffusion_scheduler.timesteps[:][step_idx]

            if self.config.use_clamp:
                x_boxes = torch.clamp(img, min=-1*self.config.norm_scale, max=1*self.config.norm_scale)
            else:
                x_boxes = img
            noisy_traj_points = self.denorm_odo(x_boxes, vec=anchor_minmax)
            # 2. proj noisy_traj_points to the query
            traj_points = torch.concat([noisy_traj_points, speed_anchor_stacked], dim=1) # torch.Size([64, 40, 8, 4])
            traj_pos_embed = gen_sineembed_for_position(traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode*2,-1)
            traj_feature, anchor_embedding = torch.chunk(traj_feature, 2, dim=1) # torch.Size([64, 20, 256]), torch.Size([64, 20, 256])


            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            if self.config.anchor_embed:
                time_embed = torch.concat([time_embed, anchor_embedding], dim=-1).to(device) 
                time_embed = self.anchor_embed(time_embed)
                if self.config.with_query_as_embedding:
                    ego_query = self.cross_attn(ego_query, agents_query, agents_query, skip=ego_query)
                    time_embed = self.fusion_embed(torch.cat([ego_query, time_embed], dim=-1))


            # 4. begin the stacked decoder
            poses_reg_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            pred_traj = torch.concatenate([
                poses_reg[...,0:2] + speed_anchor, 
                poses_reg[...,2:3]
            ],dim=-1).squeeze(1)
            traj_list.update({f"trajectory_{k}": pred_traj})

            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start, vec=anchor_minmax)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        best_traj = torch.concatenate([
                poses_reg[...,0:2] + speed_anchor, 
                poses_reg[...,2:3]
            ],dim=-1).squeeze(1)

        return {"trajectory":best_traj.squeeze(1)}
    

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        gt_trajs = targets.get('trajectory') # torch.Size([64, 8, 3])
        n_trajs = self.n_trajs
        
        plan_anchor = torch.stack([gt_trajs[...,:2]]*n_trajs, dim=1).float().to(device) # torch.Size([64, 20, 8, 2])
        odo_info_fut = self.norm_odo(plan_anchor, vec=point_minmax) # torch.Size([1280, 1, 8, 2])
        odo_info_fut = odo_info_fut.view(bs*self.n_trajs, 8, 2) # torch.Size([1280, 8, 2])
        timesteps = torch.randint(
            0, 1000,
            (bs * self.n_trajs,), device=device
        ) 
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        # import ipdb; ipdb.set_trace()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points, vec=point_minmax)
        noisy_traj_points = noisy_traj_points.view(bs, self.n_trajs, 8, 2) # torch.Size([64, 20, 8, 2])

        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64) # torch.Size([64, 20, 8, 64])
        traj_pos_embed = traj_pos_embed.flatten(-2) # torch.Size([64, 20, 8, 512])
        traj_feature = self.plan_anchor_encoder(traj_pos_embed) # torch.Size([64, 20, 256])
        traj_feature = traj_feature.view(bs,self.n_trajs,-1) # torch.Size([64, 20, 256])
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps) # torch.Size([1280, 256])
        time_embed = time_embed.view(bs,self.n_trajs,-1) # torch.Size([64, 20, 256])

        # 4. begin the stacked decoder
        poses_reg_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

        # poses_regs = torch.stack(poses_reg_list, dim=0) # torch.Size([64, 20, 8, 3])
        trajectory_loss_dict = {}
        trajectory_loss = 0.0
        for idx, pose_reg in enumerate(poses_reg_list):
            # trajectory_loss == diffusion loss, i.e, pred x0 in diffusion
            trajectory_loss_idx = F.l1_loss(pose_reg, # pred
                                            torch.stack([gt_trajs] * self.n_trajs, dim=1)# gt
                                            )
            trajectory_loss += trajectory_loss_idx
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss_idx

        return {"trajectory": poses_reg_list[-1], "trajectory_loss": trajectory_loss, "trajectory_loss_dict":trajectory_loss_dict}   

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, status_feature, global_img) -> Dict[str, torch.Tensor]:

        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(self.infer_step_num, device)
        denoise_steps = list(range(0, self.infer_step_num))

        # 1. add noise to the plan anchor
        stacked_gt_trajs = torch.randn((bs, 1, 8, 2)).to(device)
        # img = self.norm_odo(stacked_gt_trajs)
        img = self.norm_odo(stacked_gt_trajs, vec=point_minmax) 
        # noise = torch.randn(img.shape, device=device)
        # noisy_trajs = self.denorm_odo(img) 
        ego_fut_mode = img.shape[1]
        for step_idx in denoise_steps:
            k = self.diffusion_scheduler.timesteps[:][step_idx]
            x_boxes = torch.clamp(img, min=-1, max=1)
            # noisy_traj_points = self.denorm_odo(x_boxes)
            noisy_traj_points = self.denorm_odo(x_boxes, vec=point_minmax)


            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start, vec=point_minmax)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        return {"trajectory": poses_reg.squeeze(1)}
    
    