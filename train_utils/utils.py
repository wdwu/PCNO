import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_padding(x, num_pad):
    res = x.copy()
    if max(num_pad) > 0:
        for i in range(len(x)):
            res[i] = F.pad(x[i], (0, 0, num_pad[0], num_pad[1]), 'constant', 0)  # 左右
    else:
        res = x
    return res


def add_padding2(x, pad_ratio, GRID_T, GRID_X):
    NUM_PIPES = len(x)
    res = x.copy()
    if max(pad_ratio) > 0:
        num_pad1 = [round(GRID_T * pad_ratio[0])] * 2
        num_pad3 = [round(NUM_PIPES * pad_ratio[2])] * 2
        # 计算位置维度的pad
        max_length = max(GRID_X)
        num_pad2 = []
        pad2 = [round(max_length * pad_ratio[1])] * 2
        for i in range(NUM_PIPES):
            # 计算补零的长度
            pad_left = pad2[0] + round((max_length - GRID_X[i]) / 2)
            pad_right = sum(pad2) + max_length - GRID_X[i] - pad_left
            num_pad2.append([pad_left, pad_right])
            # 补零
            res[i] = F.pad(x[i], (num_pad2[i][0], num_pad2[i][1], num_pad1[0], num_pad1[1]), 'constant', 0)  # 左右上下

        # 再bus维度补零
        if max(num_pad3) > 0:
            res = torch.stack(res, dim=-1)
            res = F.pad(res, (num_pad3[0], num_pad3[1]))
            res = list(torch.unbind(res, dim=-1))

    else:
        num_pad1 = num_pad3 = [0, 0]
        num_pad2 = [[0, 0]] * NUM_PIPES
        res = x
    # num_pad2 = [[0, 0]] * NUM_PIPES
    return res, num_pad1, num_pad2, num_pad3


def remove_padding(x, num_pad):
    if max(num_pad) > 0:
        res = x[:, num_pad[0]:-num_pad[1], :, :]
    else:
        res = x
    return res


def remove_padding2(x, num_pad1, num_pad2, num_pad3):
    if max(num_pad3) > 0:
        x = x[num_pad3[0]:-num_pad3[1]]
    n = len(x)
    batchsize = x[0].shape[0]
    channel = x[0].shape[1]
    res = x.copy()
    if max(num_pad1) > 0 or max(max(num_pad2)) > 0:
        for i in range(n):
            res[i] = x[i][:, :, num_pad1[0]:-num_pad1[1], num_pad2[i][0]:-num_pad2[i][1]]
    else:
        res = x
    for j in range(n):
        res[j] = res[j].reshape(batchsize, channel, -1)
    res = torch.cat(res, dim=2).permute(0, 2, 1)
    return res


def calculate_weight(epoch, config, loss_BC, loss_IC, num_epochs_rampup=300):
    # 权重在前num_epochs_rampup个轮次内线性增加，然后保持为最大权重max_weight
    # data_w = config['train']['data_loss']
    fNS_max_weight = config['train']['fNS_loss']
    fen_max_weight = config['train']['fen_loss']
    BC_max_weight = config['train']['BC_loss']
    IC_max_weight = config['train']['IC_loss']

    if epoch < num_epochs_rampup:
        # 线性增加权重
        IC_w = IC_max_weight * ((epoch + 1) / num_epochs_rampup)
        BC_w = BC_max_weight * ((epoch + 1) / num_epochs_rampup)
    else:
        IC_w = IC_max_weight
        BC_w = BC_max_weight

    return IC_w, BC_w, fen_max_weight, fNS_max_weight


def convolution(image, difference):
    if difference == 'x':
        w = [[-1, 1], [-1, 1]]
    elif difference == 't':
        w = [[-1, -1], [1, 1]]
    elif difference == 'ones':
        w = [[1, 1], [1, 1]]
    # 定义卷积核权重
    weights = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(0)

    # 将输入图像转换为PyTorch的张量
    image_tensor = image.unsqueeze(1)

    # 创建卷积层对象
    conv_layer = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)

    # 将卷积核权重加载到卷积层
    conv_layer.weight = nn.Parameter(weights)

    # 执行卷积运算
    convolved_image = conv_layer(image_tensor)

    # 获取卷积结果
    convolved_image = convolved_image.squeeze(1)

    return convolved_image


# # 测试代码
# image = [[1, 2, 3, 4],
#          [5, 6, 7, 8],
#          [9, 10, 11, 12],
#          [13, 14, 15, 16]]
#
# convolved_image = convolution(image)
# print(convolved_image)

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class maxminnormal(object):
    def __init__(self, G_MAX, G_MIN, P_MAX, P_MIN, NUM_SOURCE, NUM_DEMAND):
        super(maxminnormal, self).__init__()

        self.Gmax = G_MAX
        self.Gmin = G_MIN

        self.Pmax = P_MAX
        self.Pmin = P_MIN

        self.minn = [self.Gmin, self.Pmin]
        self.subtrat = [(self.Gmax - self.Gmin), (self.Pmax - self.Pmin)]

        self.num_sources = NUM_SOURCE
        self.num_demand = NUM_DEMAND
        # self.eps = eps

    def encode(self, x):
        xx = x.clone()
        xx[:, :, :, 0] = (xx[:, :, :, 0] - self.Gmin) / (self.Gmax - self.Gmin)  # 流量
        xx[:, :, :, 1] = (xx[:, :, :, 1] - self.Pmin) / (self.Pmax - self.Pmin)  # 压力

        # xx[:, :, :, 1] = torch.log10(xx[:, :, :, 1]+1)/math.log10(self.Pmax)
        return xx

    def encodenew(self, x):
        x[..., -self.num_sources - self.num_demand:-self.num_sources] = (x[...,
                                                                         -self.num_sources - self.num_demand:-self.num_sources] - self.Gmin) / (
                                                                                self.Gmax - self.Gmin)  # 流量
        x[..., -self.num_sources:] = (x[..., -self.num_sources:] - self.Pmin) / (self.Pmax - self.Pmin)  # 流量
        return x

    def decode(self, x):
        xx = x.clone()
        # x is in shape of batch*n or T*batch*n
        xx[:, :, :, 0] = x[:, :, :, 0] * (self.Gmax - self.Gmin) + self.Gmin
        xx[:, :, :, 1] = x[:, :, :, 1] * (self.Pmax - self.Pmin) + self.Pmin
        # xx[:, :, :, 1] = 10**(xx[:, :, :, 1]*math.log10(self.Pmax))-1

        return xx

    def decodenew(self, x):
        # xx = x.clone()
        # x is in shape of batch*n or T*batch*n
        x[..., 0] = x[..., 0] * (self.Gmax - self.Gmin) + self.Gmin
        x[..., 1] = x[..., 1] * (self.Pmax - self.Pmin) + self.Pmin
        # xx[:, :, :, 1] = 10**(xx[:, :, :, 1]*math.log10(self.Pmax))-1

        return x

    def decodebatch(self, x):
        # xx = x.clone()
        # x is in shape of batch*n or T*batch*n
        x[..., -self.num_sources - self.num_demand:-self.num_sources] = x[...,
                                                                        -self.num_sources - self.num_demand:-self.num_sources] * (
                                                                                self.Gmax - self.Gmin) + self.Gmin  # 流量
        x[..., -self.num_sources:] = x[..., -self.num_sources:] * (self.Pmax - self.Pmin) + self.Pmin  # 流量

        # xx[:, :, :, 1] = 10**(xx[:, :, :, 1]*math.log10(self.Pmax))-1

        return x


def vor2vel(w, L=2 * np.pi):
    '''
    Convert vorticity into velocity
    Args:
        w: vorticity with shape (batchsize, num_x, num_y, num_t)

    Returns:
        ux, uy with the same shape
    '''
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 2 * np.pi / L * 1j * k_y * f_h
    uy_h = -2 * np.pi / L * 1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    return ux, uy


def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = torch.randint(s, size=(N, p))
    sample_ic_t = torch.zeros(N, p)
    sample_ic_x = index_ic / s

    # sample BC
    sample_bc = torch.rand(size=(N, p // 2))
    sample_bc_t = torch.cat([sample_bc, sample_bc], dim=1)
    sample_bc_x = torch.cat([torch.zeros(N, p // 2), torch.ones(N, p // 2)], dim=1)

    # sample I
    # sample_i_t = torch.rand(size=(N,q))
    # sample_i_t = torch.rand(size=(N,q))**2
    sample_i_t = -torch.cos(torch.rand(size=(N, q)) * np.pi / 2) + 1
    sample_i_x = torch.rand(size=(N, q))

    sample_t = torch.cat([sample_ic_t, sample_bc_t, sample_i_t], dim=1).cuda()
    sample_t.requires_grad = True
    sample_x = torch.cat([sample_ic_x, sample_bc_x, sample_i_x], dim=1).cuda()
    sample_x.requires_grad = True
    sample = torch.stack([sample_t, sample_x], dim=-1).reshape(N, (p + p + q), 2)
    return sample, sample_t, sample_x, index_ic.long()


def get_grid(N, T, s):
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = torch.tensor(np.linspace(0, 1, s + 1)[:-1], dtype=torch.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = torch.stack([gridt, gridx], dim=-1).reshape(N, T * s, 2)
    return grid, gridt, gridx


def get_2dgrid(S):
    '''
    get array of points on 2d grid in (0,1)^2
    Args:
        S: resolution

    Returns:
        points: flattened grid, ndarray (N, 2)
    '''
    xarr = np.linspace(0, 1, S)
    yarr = np.linspace(0, 1, S)
    xx, yy = np.meshgrid(xarr, yarr, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel()], axis=0).T
    return points


def torch2dgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def convert_ic(u0, N, T, time_scale=1.0):
    u0 = u0.reshape(N, T, 1).repeat([1, T, 1])

    gridt = torch.tensor(np.linspace(0, time_scale, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, 1).cuda()
    gridt.requires_grad = True

    a_data = torch.stack([gridt, u0], dim=-1)

    # gridx, gridy, gridt = get_grid(N, T, time_scale=time_scale, device=u0.device)
    # a_data = torch.cat((gridx.repeat([N, 1, 1, 1, 1]), gridy.repeat([N, 1, 1, 1, 1]),
    #                     gridt.repeat([N, 1, 1, 1, 1]), u0), dim=-1)
    return a_data


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count


def save_checkpoint(path, path1, name, model, ep, sum_loss, optimizer=None, scheduler=None):
    ckpt_dir = 'checkpoint_s/%s/%s/' % (path, path1)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    save_dict = {
        'model': model_state_dict,
        'ep': ep,
        'sum_loss': sum_loss
    }

    if optimizer is not None:
        save_dict['optim'] = optimizer.state_dict()

    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    torch.save(save_dict, ckpt_dir + name)
    # print('Checkpoint is saved at %s' % ckpt_dir + name)


def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None

    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save({
        'model': model_state,
        'optim': optim_state,
        'scheduler': scheduler_state
    }, path)
    print(f'Checkpoint is saved to {path}')


def dict2str(log_dict):
    res = ''
    for key, value in log_dict.items():
        res += f'{key}: {value}|'
    return res


class GaussianRF(object):
    def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False,
                 device=None):
        '''
        size 被用来构造一个包含了正负半轴长度的波数序列 k。当生成样本的长度 size 是奇数时，由于存在一个正负波数，会导致波数序列 k 中的 0 波数出现两次。
        这个问题会在生成波数序列 k 的过程中引起重复，从而影响到生成的高斯随机场的频谱特性。
        为了避免这种情况，一般建议将样本的长度 size 设置为偶数，这样可以确保波数序列 k 中不会出现重复的 0 波数，使得高斯随机场的频谱特性更加准确。


        alpha 参数是高斯随机场的光滑度参数，控制了高斯随机场的空间平滑度。
        具体来说，alpha 越小，高斯随机场的变化越剧烈，其空间波动性越强；alpha 越大，高斯随机场的变化越平缓，其空间波动性越弱。

        sigma 参数是高斯随机场的标准差参数，影响了高斯随机场的振幅大小。
        具体来说，sigma 越大，生成的高斯随机场的振幅越大；sigma 越小，生成的高斯随机场的振幅越小。

        tau 参数是高斯随机场的特征时间尺度，通常用于控制高斯随机场的频率特性。
        具体来说，tau 越小，意味着高频成分的权重越高，生成的高斯随机场的空间波动性越强；tau 越大，意味着低频成分的权重越高，生成的高斯随机场的空间波动性越弱。
        在给定的代码中，sigma 参数的计算与 tau 相关，具体来说，sigma 的计算使用了 tau 的平方根，因此 tau 对 sigma 的选择也会影响到生成的高斯随机场的振幅大小。

        length 参数代表了高斯随机场的长度尺度。这个长度尺度用于计算高斯随机场的方差，并影响高斯随机场的空间特征。
        而 length 参数则通过这个公式中的 τ 来影响方差。通常，length 越大，方差越大，高斯随机场的空间波动性越强，变化越平缓；而 length 越小，方差越小，高斯随机场的空间波动性越弱，变化越剧烈。
        '''

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        const = (4 * (math.pi ** 2)) / (length ** 2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((const * (k ** 2) + tau ** 2) ** (-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0] = size * sigma * (tau ** (-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                    (const * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0, 0] = (size ** 2) * sigma * (tau ** (-alpha))
            else:
                self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                    (const * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))

            if constant_eig:
                self.sqrt_eig[0, 0, 0] = (size ** 3) * sigma * (tau ** (-alpha))
            else:
                self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def custom_sigmoid(self, x, scale=1.0):
        """自定义的Sigmoid函数，通过'scale'参数调整灵敏度"""
        return 1 / (1 + torch.exp(-x * scale))

    def sample(self, N, value_range=None, sigmoid_scale=1):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")

        # 如果提供了 value_range，则使用Sigmoid变换调整样本值
        u = (u - u.mean()) / u.std()
        if value_range is not None:
            a, b = value_range
            u = self.custom_sigmoid(u, scale=sigmoid_scale)  # Sigmoid变换到(0, 1)
            u = a + (b - a) * u  # 缩放和位移调整到(a, b)
        return u
