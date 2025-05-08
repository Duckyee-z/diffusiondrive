import torch


traj_point_stats = torch.tensor([
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

traj_point_minmax_stats = torch.tensor([
    # Point 0
    [[-0.40, 8.58],   # x: min=-0.40, max=8.58
     [-0.40, 0.98]],  # y: min=-0.40, max=0.98
    # Point 1
    [[-0.77, 16.63],[-1.62, 2.16]],
    # Point 2
    [[-1.12, 24.85],[-3.43, 3.96]],
    # Point 3
    [[-1.39, 33.01],[-5.85, 6.59]],
    # Point 4
    [[-1.53, 41.13],[-8.59, 9.93]],
    # Point 5
    [[-1.57, 49.10],[-11.95, 13.63]],
    # Point 6
    [[-1.55, 57.15],[-15.71, 17.83]],
    # Point 7
    [[-1.38, 65.17],[-19.68, 22.32]]
], dtype=torch.float32)


def norm_traj(traj, type='znorm'): # traj_info_fut ([64, 20, 8, 2])
    if type == 'znorm':
        return norm_traj_znorm(traj, stats=traj_point_stats)
    elif type == 'minmax':
        return norm_traj_minmax(traj, vec=traj_point_minmax_stats)
    else :
        raise ValueError("Unknown normalization type: {}".format(type))
    
def denorm_traj(traj, type='znorm'):
    if type == 'znorm':
        return denorm_traj_znorm(traj, stats=traj_point_stats)
    elif type == 'minmax':
        return denorm_traj_minmax(traj, vec=traj_point_minmax_stats)
    else :
        raise ValueError("Unknown denormalization type: {}".format(type))

def norm_traj_znorm(odo_info_fut, stats=traj_point_stats): # odo_info_fut ([64, 20, 8, 2])
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

def denorm_traj_znorm( odo_info_fut, stats=traj_point_stats):
    # 提取均值和标准差，形状均为(8, 2)
    stats.to(odo_info_fut.device)
    mean = stats[:, :, 0].to(odo_info_fut.device)
    std = stats[:, :, 1].to(odo_info_fut.device)
    
    # 调整形状以便广播，(1, 1, 8, 2)
    mean = mean.unsqueeze(0).unsqueeze(0)
    std = std.unsqueeze(0).unsqueeze(0)

    denormed_odo_info_fut = odo_info_fut * std + mean

    return denormed_odo_info_fut

def norm_traj_minmax(odo_info_fut, vec=traj_point_minmax_stats): # odo_info_fut ([64, 20, 8, 2])
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
        normalized_x = 2* (x_coords - x_mins[None, None, :]) / x_range[None, None, :] - 1
        
        y_range = y_maxs - y_mins
        normalized_y = 2* (y_coords - y_mins[None, None, :]) / y_range[None, None, :] - 1 
        
        # 合并结果并保持原有维度

        return torch.stack([normalized_x, normalized_y], dim=-1)
    
def denorm_traj_minmax(odo_info_fut, vec=traj_point_minmax_stats):
        # 提取每个点的x和y的min、max
        x_mins = vec[:, 0, 0].to(odo_info_fut.device)  # 形状 (8,)
        x_maxs = vec[:, 0, 1].to(odo_info_fut.device)
        y_mins = vec[:, 1, 0].to(odo_info_fut.device)
        y_maxs = vec[:, 1, 1].to(odo_info_fut.device)

        # 分离x和y坐标
        x_norm = odo_info_fut[..., 0]  # 形状 (B, L, 8)
        y_norm = odo_info_fut[..., 1]

        # 计算归一化后的坐标，利用广播机制
        x_range = x_maxs - x_mins
        denormalized_x = ((x_norm + 1) / 2) * x_range[None, None, :] + x_mins[None, None, :]
        
        y_range = y_maxs - y_mins
        denormalized_y = ((y_norm + 1) / 2) * y_range[None, None, :] + y_mins[None, None, :]
        
        return torch.stack([denormalized_x, denormalized_y], dim=-1)

