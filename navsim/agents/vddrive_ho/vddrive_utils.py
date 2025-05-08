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

acc_point_stats = torch.tensor([
    # 第1个样本
    [[-0.0399, 0.8115], [0.1323, 0.6393]], 
    # 第2个样本
    [[-0.0219, 0.8488], [0.1327, 0.6353]], 
    [[-0.0103, 0.8691], [0.1293, 0.6300]], 
    [[-0.0080, 0.8693], [0.1206, 0.6248]], 
    [[-0.0137, 0.8507], [0.1110, 0.6169]], 
    [[-0.0217, 0.8187], [0.0995, 0.6049]], 
    [[-0.0301, 0.7786], [0.0887, 0.5867]], 
    [[-0.0356, 0.7352], [0.0791, 0.5686]]

], dtype=torch.float32)

velo_point_stats = torch.tensor([
    [[4.5858, 3.1120], [-0.0565, 0.0906]],   # Row 0: [mu_x, sigma_x], [mu_y, sigma_y]
    [[4.5711, 3.0297], [-0.0556, 0.0897]],   # Row 1
    [[4.5636, 2.9677], [-0.0546, 0.0891]],   # Row 2
    [[4.5586, 2.9384], [-0.0537, 0.0891]],   # Row 3
    [[4.5511, 2.9489], [-0.0526, 0.0894]],   # Row 4
    [[4.5388, 2.9968], [-0.0516, 0.0900]],   # Row 5
    [[4.5216, 3.0729], [-0.0507, 0.0909]],   # Row 6
    [[4.5006, 3.1666], [-0.0499, 0.0918]]    # Row 7
], dtype=torch.float32)


acc_point_minmax_stats = torch.tensor([
    # 第1个样本
    [[-2.47, 2.39], [-1.79, 2.05]], 
    # 第2个样本
    [[-2.57, 2.52], [-1.77, 2.04]], 
    [[-2.62, 2.6], [-1.76, 2.02]], 
    [[-2.62, 2.6], [-1.75, 2.0]], 
    [[-2.57, 2.54], [-1.74, 1.96]], 
    [[-2.48, 2.43], [-1.72, 1.91]], 
    [[-2.37, 2.31], [-1.67, 1.85]], 
    [[-2.24, 2.17], [-1.63, 1.79]]

], dtype=torch.float32)




def norm_traj(traj, type='znorm'): # traj_info_fut ([64, 20, 8, 2])
    if type == 'znorm':
        return norm_traj_znorm(traj, stats=traj_point_stats)
    elif type == 'minmax':
        raise NotImplementedError("minmax normalization is not implemented")
    else :
        raise ValueError("Unknown normalization type: {}".format(type))
    
def denorm_traj(traj, type='znorm'):
    if type == 'znorm':
        return denorm_traj_znorm(traj, stats=traj_point_stats)
    elif type == 'minmax':
        raise NotImplementedError("minmax denormalization is not implemented")
    else :
        raise ValueError("Unknown denormalization type: {}".format(type))

def norm_acc(acc, type='znorm'): # traj_info_fut ([64, 20, 8, 2])
    if type == 'znorm':
        return norm_traj_znorm(acc, stats=acc_point_stats)
    elif type == 'minmax':
        return norm_traj_minmax(acc, vec=acc_point_minmax_stats)
    else :
        raise ValueError("Unknown normalization type: {}".format(type))
def denorm_acc(acc, type='znorm'):
    if type == 'znorm':
        return denorm_traj_znorm(acc, stats=acc_point_stats)
    elif type == 'minmax':
        return denorm_traj_minmax(acc, vec=acc_point_minmax_stats)
    else :
        raise ValueError("Unknown denormalization type: {}".format(type))
def norm_velo(velo, type='znorm'): # traj_info_fut ([64, 20, 8, 2])
    if type == 'znorm':
        return norm_traj_znorm(velo, stats=velo_point_stats)
    elif type == 'minmax':
        raise NotImplementedError("minmax normalization is not implemented")
    else :
        raise ValueError("Unknown normalization type: {}".format(type))
def denorm_velo(velo, type='znorm'):
    if type == 'znorm':
        return denorm_traj_znorm(velo, stats=velo_point_stats)
    elif type == 'minmax':
        raise NotImplementedError("minmax denormalization is not implemented")
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

def norm_traj_minmax(odo_info_fut, vec=acc_point_minmax_stats): # odo_info_fut ([64, 20, 8, 2])
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
    
def denorm_traj_minmax(odo_info_fut, vec=acc_point_minmax_stats):
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
        velocities: Tensor of shape [B, L, T, 2] containing vx, vy velocity components
        initial_position: Optional initial positions of shape [B, 2]. If None, assumes zeros
        initial_velocity: Optional initial velocities of shape [B, 2]. If None, assumes zeros
        dt: Time step between samples (default: 0.1s)
        order: Integration order (1: velocity to position, 2: acceleration to position)
        
    Returns:
        Tensor of shape [B, T, 2] containing x, y position coordinates
    """
    batch_size, N, time_steps, dims = velocities.shape
    device = velocities.device

    velocities = velocities.view(batch_size * N, time_steps, dims)  # [B*N, T, 2]
    if initial_velocity is not None:
        initial_velocity = initial_velocity.unsqueeze(1).repeat(1, N, 1)  # [B*N, 2]
        initial_velocity = initial_velocity.view(batch_size * N, dims)  # [B*N, 2]
    
    # 高阶积分处理 (从加速度到速度)
    if order == 2:
        # 首先将加速度积分为速度
        velocities_from_accel = high_order_integration(velocities, initial_velocity, dt)
        velocities_from_accel = velocities_from_accel.view(batch_size, N, time_steps, dims)
        # 然后将速度积分为位置
        return high_order_to_trajectory(velocities_from_accel, initial_position, None, dt, order=1).view(batch_size, N, time_steps, dims)
    
    # 处理初始位置
    if initial_position is None:
        initial_position = torch.zeros(batch_size* N, dims, device=device)
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
    
    return trajectories.view(batch_size, N, time_steps, dims)
