from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.vdiffusiondrivev2.transfuser_config import TransfuserConfig
from navsim.agents.vdiffusiondrivev2.transfuser_backbone import TransfuserBackbone
from navsim.agents.vdiffusiondrivev2.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.vdiffusiondrivev2.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.vdiffusiondrivev2.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.vdiffusiondrivev2.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union


point_stats = torch.tensor([
    # 点0 (x,y) (μ, σ)
    [[2.24, 1.53], [0.01, 0.07]],
    # 点1
    [[4.46, 3.01], [0.06, 0.28]],
    # 点2
    [[6.67, 4.44], [0.14, 0.63]],
    # 点3
    [[8.85, 5.81], [0.24, 1.11]],
    # 点4
    [[10.99, 7.15], [0.38, 1.70]],
    # 点5
    [[13.10, 8.46], [0.53, 2.41]],
    # 点6
    [[15.17, 9.77], [0.71, 3.21]],
    # 点7
    [[17.19, 11.09], [0.90, 4.10]]
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
        time_embed,
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
                 config,
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
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
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
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
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
            poses_reg = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
    
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list

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
        print("================= Init vdiffusiondrivev2.3 ====================")
        self.n_trajs = 20

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.infer_step_num = config.infer_step_num
        self.trailing_mode = config.trailing_mode
        self.zero_infer = config.zero_infer

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

        self.loss_computer = LossComputer(config)

    def norm_odo(self, odo_info_fut, stats=point_stats): # odo_info_fut ([64, 20, 8, 2])
        """
        Args:
            tensor: 输入张量，形状为(B, L, 8, 2)
            stats: 统计量张量，形状为(8, 2, 2), 其中stats[i, j, 0]为第i个点第j坐标的均值, stats[i, j, 1]为标准差
        Returns:
            normalized_tensor: 标准化后的张量，形状与输入相同
        """
        # 提取均值和标准差，形状均为(8, 2)
        stats.to(odo_info_fut.device)
        mean = stats[:, :, 0].to(odo_info_fut.device)
        std = stats[:, :, 1].to(odo_info_fut.device)
        
        # 调整形状(1, 1, 8, 2)
        mean = mean.unsqueeze(0).unsqueeze(0)
        std = std.unsqueeze(0).unsqueeze(0)
        
        # 执行标准化
        normed_odo_info_fut = (odo_info_fut - mean) / std

        return normed_odo_info_fut
    
    def denorm_odo(self, odo_info_fut, stats=point_stats):
        # 提取均值和标准差，形状均为(8, 2)
        stats.to(odo_info_fut.device)
        mean = stats[:, :, 0].to(odo_info_fut.device)
        std = stats[:, :, 1].to(odo_info_fut.device)
        
        # 调整形状以便广播，(1, 1, 8, 2)
        mean = mean.unsqueeze(0).unsqueeze(0)
        std = std.unsqueeze(0).unsqueeze(0)

        denormed_odo_info_fut = odo_info_fut * std + mean

        return denormed_odo_info_fut
    
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, status_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape, status_encoding, status_feature, targets, global_img)
        else:
            if self.trailing_mode:
                return self.forward_test_trailing_mode(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature,global_img)
            return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, global_img)


    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, status_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        gt_trajs = targets.get('trajectory') # torch.Size([64, 8, 3])
        velocity = targets.get('velocity') # torch.Size([64, 8, 2])
        acceleration = targets.get('acceleration') # torch.Size([64, 8, 2])

        # status_feature [driving_command, ego_velocity, ego_acceleration] = [_, _, _, _, vx, vy, ax, ay]
        velo_now = status_feature[..., 4:6] # torch.Size([64, 2])
        acc_now = status_feature[..., 6:8] # torch.Size([64, 2])
        
        stacked_gt_trajs = gt_trajs[...,:2].unsqueeze(1).repeat(1, self.n_trajs, 1, 1).reshape(bs * self.n_trajs, 8, 2).to(device)
        stacked_gt_trajs = stacked_gt_trajs.unsqueeze(1) # torch.Size([1280, 1, 8, 2])
        odo_info_fut = self.norm_odo(stacked_gt_trajs) # torch.Size([1280, 1, 8, 2])
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
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        noisy_traj_points = noisy_traj_points.view(bs, self.n_trajs, 8, 2) # torch.Size([64, 20, 8, 2])

        ego_fut_mode = noisy_traj_points.shape[1] # ego_fut_mode = self.n_trajs
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64) # torch.Size([64, 20, 8, 64])
        traj_pos_embed = traj_pos_embed.flatten(-2) # torch.Size([64, 20, 8, 512])
        traj_feature = self.plan_anchor_encoder(traj_pos_embed) # torch.Size([64, 20, 256])
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1) # torch.Size([64, 20, 256])
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps) # torch.Size([1280, 256])
        time_embed = time_embed.view(bs,self.n_trajs,-1) # torch.Size([64, 1, 256])

        # 4. begin the stacked decoder
        poses_reg_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

        # poses_regs = torch.stack(poses_reg_list, dim=0) # torch.Size([64, 20, 8, 3])
        trajectory_loss_dict = {}
        trajectory_loss = 0
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
        # n_trajs = self.n_trajs
        # self.diffusion_scheduler.set_timesteps(1000, device)

        self.diffusion_scheduler.set_timesteps(self.infer_step_num, device)
        denoise_steps = list(range(0, self.infer_step_num))

        # 1. add noise to the plan anchor
        stacked_gt_trajs = torch.randn((bs, 1, 8, 2)).to(device)
        img = self.norm_odo(stacked_gt_trajs)
        # noise = torch.randn(img.shape, device=device)
        # noisy_trajs = self.denorm_odo(img) 
        ego_fut_mode = img.shape[1]
        for step_idx in denoise_steps:
            k = self.diffusion_scheduler.timesteps[:][step_idx]
            
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

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
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        return {"trajectory": poses_reg.squeeze(1)}
    
    
    def forward_test_trailing_mode(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # n_trajs = self.n_trajs
        # self.diffusion_scheduler.set_timesteps(1000, device)

        self.diffusion_scheduler.set_timesteps(self.infer_step_num, device)
        times = torch.linspace(-1, 1000-1, steps=self.infer_step_num + 1)
        times = list(reversed(times.int().tolist()))
        sample_timesteps = torch.tensor(times).to(device)

        # 1. add noise to the plan anchor
        stacked_gt_trajs = torch.randn((bs, 1, 8, 2)).to(device)
        if self.zero_infer:
            stacked_gt_trajs = torch.zeros((bs, 1, 8, 2)).to(device)
        img = self.norm_odo(stacked_gt_trajs)
        # noise = torch.randn(img.shape, device=device)
        # noisy_trajs = self.denorm_odo(img) 
        ego_fut_mode = img.shape[1]
        for k in sample_timesteps[:]:
            if k == -1 : continue
            
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

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
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        return {"trajectory": poses_reg.squeeze(1)}
    

    
