# from utilities3 import GaussianRF, save_checkpoint
import os
from timeit import default_timer
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import Sampler
from train_utils.adam import Adam



import logging
from datetime import datetime
import re


#############################################################
# V61版本 融合仿真数据和采样数据训练。
#############################################################
def setup_logging(current_time, log_dir='checkpoint_s/log/', log_prefix='training'):
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.basename(__file__)[4:7]  # 获取文件名的一部分
    log_filename = f'{file_name}_{log_prefix}_{current_time}.log'
    log_filepath = os.path.join(log_dir, log_filename)
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 配置日志记录器
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )



###### 数据处理 ######################################################
# 时域两端补零
def add_padding2(x, pad_ratio, GRID_X):
    GRID_T = x[0].size(2)
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


def remove_padding2(x, num_pad1, num_pad2, num_pad3):
    if max(num_pad3) > 0:
        x = x[num_pad3[0]:-num_pad3[1]]
    n = len(x)
    batchsize = x[0].shape[0]
    channel = x[0].shape[1]
    res = x.copy()
    if max(num_pad1) > 0 or max(max(num_pad2)) > 0:
        for i in range(n):
            res[i] = x[i][:, :, num_pad1[0]:x[i].size(2) - num_pad1[1], num_pad2[i][0]:x[i].size(3) - num_pad2[i][1]]
    else:
        res = x
    res = torch.cat(res, dim=-1).permute(0, 2, 3, 1)
    return res


# 保存检查点
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


# 加载配置文件
def load_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


# 提取数据
def pull_data(loaded_data_list):
    keys_to_extract = [
        'gas_net', 'pipe_lable', 'map_edge_name', 'EDGE_INDEX_PIPES', 'NUM_NODES',
        'NUM_PIPES', 'NUM_SOURCE', 'NUM_DEMAND', 'source_nodes', 'demand_nodes',
        'G_MAX', 'G_MIN', 'P_MAX', 'P_MIN', 'PIPES_LEN', 'DIAMETER', 'Z_FACTOR',
        'R_FACTOR', 'T_FACTOR', 'FRICTION', 'Sample_nt', 'time_interval',
        'SUB_X', 'SUB_T', 'GRID_X', 'GRID_T', 'x_l', 'num_merged_nodes',
        'node_stage',
    ]
    # 提取数据
    data = {key: loaded_data_list[key] for key in keys_to_extract}
    return data


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
        return xx

    def encodenew(self, x):
        x[..., -self.num_sources - self.num_demand:-self.num_sources] = (x[..., -self.num_sources - self.num_demand:-self.num_sources] - self.Gmin) / (
                self.Gmax - self.Gmin)  # 流量
        x[..., -self.num_sources:] = (x[..., -self.num_sources:] - self.Pmin) / (self.Pmax - self.Pmin)  # 流量
        return x

    def decodenew(self, x):
        original_sizes = [tensor.size(2) for tensor in x]
        x = torch.cat(x, dim=2)
        x[..., 0] = x[..., 0] * (self.Gmax - self.Gmin) + self.Gmin
        x[..., 1] = x[..., 1] * (self.Pmax - self.Pmin) + self.Pmin
        x = list(torch.split(x, original_sizes, dim=2))
        return x

    def decodebatch(self, x):
        x[..., -self.num_sources - self.num_demand:-self.num_sources] = x[..., -self.num_sources - self.num_demand:-self.num_sources] * (
                self.Gmax - self.Gmin) + self.Gmin  # 流量
        x[..., -self.num_sources:] = x[..., -self.num_sources:] * (self.Pmax - self.Pmin) + self.Pmin  # 流量

        return x


# 加载仿真数据文件
def load_simulation_data(num_sim, file_path_template, subt=None, subx=None):
    data_keys = ['Pressure', 'Flow', 'Density', 'Velocity']
    simulations = {key: [] for key in data_keys}

    for i in range(num_sim):
        sequence = f'{i:04d}'
        if subx is None:
            file_path = file_path_template.format(sequence)
        else:
            file_path = file_path_template.format(subt, subx, subt, subx, sequence)
        new_dict = np.load(file_path, allow_pickle=True).item()

        for key in data_keys:
            simulations[key].append(new_dict[key])

    for key in data_keys:
        simulations[key] = [torch.stack([torch.tensor(simulations[key][i][j]) for i in range(num_sim)], axis=0)
                            for j in range(3)]

    return simulations


def relative_error(x, y, epsilon=1e-16):
    # 计算绝对误差
    absolute_error = torch.abs(x - y)
    # 计算相对误差
    relative_error = absolute_error / (torch.abs(y) + epsilon)
    return relative_error


def rmse_loss(y_pred, y, epsilon=1e-16):
    mse = F.mse_loss(y_pred, y)
    return torch.sqrt(mse + epsilon)


def normalized_mse(y_pred, y, label, gdata):
    # 计算归一化均方误差（NMSE）。
    if label == 'flow':
        maxv, minv = gdata.data['G_MAX'], gdata.data['G_MIN']
    elif label == 'pressure':
        maxv, minv = gdata.data['P_MAX'], gdata.data['P_MIN']
    else:
        raise ValueError("Unsupported label. Choose 'flow' or 'pressure'.")
    mse = F.mse_loss(y_pred, y)
    range_square = (maxv - minv) ** 2
    nmse = mse / range_square
    return nmse



class Graph_SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(Graph_SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = 3
        self.w_conv2d = nn.Conv2d(in_channels, out_channels, 1)
        self.w_conv2d2 = nn.Conv2d(in_channels, out_channels, 1)
        self.act = _get_act('relu', 'complex')
        self.act_w = _get_act('relu', 'real')
        self.scale = (1 / (in_channels * out_channels))
        # self.scale = torch.sqrt(torch.tensor(2.0 / out_channels))
        if method in [26]:
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights3 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        if method in [34]:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        if method in [25]:
            self.weights1 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, self.modes3, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, self.modes3, self.modes1, self.modes2, dtype=torch.cfloat))
        if method in [35,39]:
            self.weights10 = nn.Parameter(self.scale * torch.rand(3, 3, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights11 = nn.Parameter(self.scale * torch.rand(3, 3, self.modes1, self.modes2, dtype=torch.cfloat))
        if method in [35]:
            self.weights12 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, dtype=torch.cfloat))
            self.weights13 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, dtype=torch.cfloat))
        if method in [8, 38, 39]:
            self.weights15 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, 3, dtype=torch.cfloat))
            self.weights16 = nn.Parameter(self.scale * torch.rand(out_channels, out_channels, 3, dtype=torch.cfloat))



    def forward(self, x, method, num_pad1, num_pad2, num_pad3, kernel_low=None, kernel_high=None):
        batchsize = x[0].size(0)
        FNO_in = []
        FNO_in_h = []

        for n in range(len(x)):
            x_ft = torch.fft.rfftn(x[n], dim=[2, 3])
            if method in [1, 2, 3, 6, 8, 12, 13, 15, 19, 20, 22, 23, 24, 25, 27, 32, 35, 39]:
                FNO_in.append(x_ft[:, :, :self.modes1, :self.modes2].unsqueeze(2))
                FNO_in_h.append(x_ft[:, :, -self.modes1:, :self.modes2].unsqueeze(2))
        if method == 25:
            # 25. 对不同通道卷积，对应实验中的FNO2D
            FNO_in = torch.cat(FNO_in, dim=2)
            FNO_in_h = torch.cat(FNO_in_h, dim=2)
            FNO_out = torch.einsum("bizxy,iozxy->bozxy", FNO_in, self.weights1)
            FNO_out_h = torch.einsum("bizxy,iozxy->bozxy", FNO_in_h, self.weights4)
            x_pipe_out_2 = [torch.zeros(batchsize, self.out_channels, x[0].size(2), x[i].size(3), device=x[0].device, dtype=torch.cfloat) for i in range(3)]
            out = [[] for _ in range(3)]
            for i in range(3):
                x_pipe_out_2[i][:, :, :self.modes1, :self.modes2] = FNO_out[:, :, i]
                x_pipe_out_2[i][:, :, -self.modes1:, :self.modes2] = FNO_out_h[:, :, i]
                out[i] = torch.fft.irfftn(x_pipe_out_2[i], s=(x[0].size(2), x[i].size(3)), dim=[2, 3])
        elif method == 26:
            # 26. 在FNO-3D
            x = torch.stack(x, dim=-1)
            x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
            z_dim = min(x_ft.shape[4], self.modes3)

            out = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), device=x.device, dtype=torch.cfloat)
            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
            out[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights1)

            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
            out[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights2)
            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :z_dim]
            out[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights3)
            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :z_dim]
            out[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights4)
            out = torch.fft.irfftn(out, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1])
            out = list(torch.unbind(out, dim=-1))
        elif method == 34:
            # PCNO-3D
            x = torch.stack(x, dim=-1)
            x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
            z_dim = min(x_ft.shape[4], self.modes3)
            out = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), device=x.device, dtype=torch.cfloat)
            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
            out[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights1)

            coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)
            coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
            out[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, self.weights2)
            out = torch.fft.irfftn(out, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1])
            out = list(torch.unbind(out, dim=-1))
        elif method == 35:
            # PCNO-C
            FNO_in = torch.cat(FNO_in, dim=2)
            FNO_in_h = torch.cat(FNO_in_h, dim=2)
            FNO_out = torch.einsum("bizxy,zoxy->bioxy", FNO_in, self.weights10)
            FNO_out_h = torch.einsum("bizxy,zoxy->bioxy", FNO_in_h, self.weights11)
            FNO_out = torch.einsum("bizxy,io->bozxy", FNO_out, self.weights12)
            FNO_out_h = torch.einsum("bizxy,io->bozxy", FNO_out_h, self.weights13)
            x_pipe_out_2 = [torch.zeros(batchsize, self.out_channels, x[0].size(2), x[i].size(3), device=x[0].device, dtype=torch.cfloat) for i in range(3)]
            out = [[] for _ in range(3)]
            for i in range(3):
                x_pipe_out_2[i][:, :, :self.modes1, :self.modes2] = FNO_out[:, :, i]
                x_pipe_out_2[i][:, :, -self.modes1:, :self.modes2] = FNO_out_h[:, :, i]
                out[i] = torch.fft.irfftn(x_pipe_out_2[i], s=(x[0].size(2), x[i].size(3)), dim=[2, 3])
        elif method == 39:
            # PCNO
            FNO_in = torch.cat(FNO_in, dim=2)
            FNO_in_h = torch.cat(FNO_in_h, dim=2)
            FNO_out = torch.einsum("bizxy,zoxy->bioxy", FNO_in, self.weights10)
            FNO_out_h = torch.einsum("bizxy,zoxy->bioxy", FNO_in_h, self.weights11)
            FNO_out = torch.einsum("bizxy,ioz->bozxy", FNO_out, self.weights15)
            FNO_out_h = torch.einsum("bizxy,ioz->bozxy", FNO_out_h, self.weights16)

            x_pipe_out_2 = [torch.zeros(batchsize, self.out_channels, x[0].size(2), x[i].size(3), device=x[0].device, dtype=torch.cfloat) for i in range(3)]
            out = [[] for _ in range(3)]
            for i in range(3):
                x_pipe_out_2[i][:, :, :self.modes1, :self.modes2] = FNO_out[:, :, i]
                x_pipe_out_2[i][:, :, -self.modes1:, :self.modes2] = FNO_out_h[:, :, i]
                out[i] = torch.fft.irfftn(x_pipe_out_2[i], s=(x[0].size(2), x[i].size(3)), dim=[2, 3])

        ########################################################################################################################
        return out


class _get_act():
    def __init__(self, act, modes):
        self.modes = modes
        if act == 'tanh':
            self.func = F.tanh
        elif act == 'gelu':
            self.func = F.gelu
        elif act == 'relu':
            self.func = F.relu_
        elif act == 'elu':
            self.func = F.elu_
        elif act == 'leaky_relu':
            self.func = F.leaky_relu_
        elif act == 'sigmoid':
            self.func = F.sigmoid
        else:
            raise ValueError(f'{act} is not supported')

    def __call__(self, input):
        if self.modes == 'real':
            return self.func(input)
        elif self.modes == 'complex':  # can‘t use relu_ that is in-place version of relu
            return F.gelu(input.real).type(torch.complex64) + 1j * F.gelu(input.imag).type(torch.complex64)





class GNO2d(nn.Module):
    def __init__(self, modes1, modes2, fc_dim=128, layers=None, in_dim=4, out_dim=2, act='gelu',
                 pad_ratio=[0., 0., 0.]):
        super(GNO2d, self).__init__()
        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.out_dim = out_dim
        self.layers = layers

        self.fc0 = nn.Linear(in_dim, layers[0])
        self.graph_convs = nn.ModuleList([Graph_SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])
        self.ws = nn.ModuleList(
            nn.Conv2d(in_size, out_size, 1) for in_size, out_size in zip(self.layers, self.layers[1:]))
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act, 'real')

    def forward(self, x, method, kernel_low=None, kernel_high=None):
        original_sizes = [tensor.size(2) for tensor in x]
        n_pipes = len(x)
        x = torch.cat(x, dim=2)
        x = self.fc0(x).permute(0, 3, 1, 2)
        len_layers = len(self.graph_convs)
        x = list(torch.split(x, original_sizes, dim=3))
        x, num_pad1, num_pad2, num_pad3 = add_padding2(x, self.pad_ratio, original_sizes)

        for i, (speconv, w) in enumerate(zip(self.graph_convs, self.ws)):
            if kernel_low == None:
                x1 = speconv(x, method, num_pad1, num_pad2, num_pad3)  # FFT
                for j in range(3):
                    x[j] = x1[j] + w(x[j])
            else:
                x = speconv(x, method, self.pad_ratio, kernel_low[i], kernel_high[i])  # FFT
            if i != len_layers - 1:
                for j in range(n_pipes):
                    x[j] = self.act(x[j])

        x = remove_padding2(x, num_pad1, num_pad2, num_pad3)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        # x = self.act(x)  # 另外加的，加了后loss降得慢
        if x.shape[-1] != self.out_dim:
            raise ValueError('The dimension of output is not match')
        x = list(torch.split(x, original_sizes, dim=2))
        return x

#####train############################################################
def load_checkpoint(model, optimizer, scheduler, checkpoint_path="None"):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)  # 加载 checkpoint
        model.load_state_dict(checkpoint['model'])  # 加载模型权重
        if optimizer is not None and 'optim' in checkpoint:
            optimizer.load_state_dict(checkpoint['optim'])  # 加载优化器状态

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        # 提取时间字符串
        current_time_match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', checkpoint_path)
        current_time = current_time_match.group() if current_time_match else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return checkpoint['ep'] + 1, current_time
    else:
        return 0, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convolution(image, difference, theta=0.8):
    if difference == 'x':
        w = [[-(1 - theta), (1 - theta)], [-theta, theta]]
    elif difference == 't':
        w = [[-1 / 2, -1 / 2], [1 / 2, 1 / 2]]
    elif difference == 'ones':
        w = [[(1 - theta) / 2, (1 - theta) / 2], [theta / 2, theta / 2]]
    # 定义卷积核权重
    weights = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
    # 创建卷积层对象
    conv_layer = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
    # 将卷积核权重加载到卷积层
    conv_layer.weight = nn.Parameter(weights)

    num_dimensions = image.dim()

    if num_dimensions == 3:
        # 将输入图像转换为PyTorch的张量
        image_tensor = image.unsqueeze(1)

        # 执行卷积运算
        convolved_image = conv_layer(image_tensor)

        # 获取卷积结果
        convolved_image = convolved_image.squeeze(1)
    else:
        # 将输入图像转换为PyTorch的张量
        image_tensor = image.unsqueeze(0).unsqueeze(0)

        # 执行卷积运算
        convolved_image = conv_layer(image_tensor)

        # 获取卷积结果
        convolved_image = convolved_image.squeeze(0).squeeze(0)

    return convolved_image


def simple_collate_fn(batch):
    return batch  # 直接返回批次数据，不进行任何处理


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(dataset)

    def __iter__(self):
        # 生成随机索引
        if self.shuffle:
            indices1 = torch.randperm(self.dataset.dataset1_len).tolist()
            indices2 = torch.randperm(self.dataset.dataset2_len).tolist()
        else:
            # # 生成按顺序排列的索引
            indices1 = list(range(self.dataset.dataset1_len))
            indices2 = list(range(self.dataset.dataset2_len))
        batches = []

        # 按 batch_size 切分索引，并生成批次
        for i in range(0, self.dataset.dataset1_len, self.batch_size):
            batches.append((1, indices1[i:i + self.batch_size]))

        for i in range(0, self.dataset.dataset2_len, self.batch_size):
            batches.append((2, indices2[i:i + self.batch_size]))

        # 根据 shuffle 参数决定是否打乱
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return len(list(range(0, self.dataset.dataset1_len, self.batch_size))) + len(list(range(0, self.dataset.dataset2_len, self.batch_size)))


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, labels):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.labels = labels  # labels 是一个包含多个字典的结构
        self.labels = labels
        if dataset1 is None:
            self.dataset1_len = 0
        else:
            self.dataset1_len = len(dataset1[0])
        self.dataset2_len = len(dataset2[0])

    def __len__(self):
        return self.dataset1_len + self.dataset2_len

    def __getitems__(self, idx):
        # 返回每个张量中对应索引的数据，如果有标签也返回标签
        if idx[0] == 1:
            batch_data = [tensor[idx[1]] for tensor in self.dataset1]
            batch_labels = None
            return batch_data, batch_labels
        # 否则获取 dataset2 的数据
        else:
            batch_data = [tensor[idx[1]] for tensor in self.dataset2]
            batch_labels = {key: [tensor[idx[1]] for tensor in tensor_list] for key, tensor_list in self.labels.items()}
            return batch_data, batch_labels


def PINO_loss(uu, x_batch, labels=None, data_dict1=None, data_dict2=None, data_dict3=None):
    if data_dict3 is not None:
        data = data_dict3
    elif labels is None:
        data = pull_data(data_dict1)
    else:
        data = pull_data(data_dict2)
    u = uu.copy()  # 防止在利用输出计算loss时改变了输出结果，从而无法继续backward
    batch = x_batch.copy()

    # 解码
    maxmin_normal = maxminnormal(data['G_MAX'], data['G_MIN'], data['P_MAX'], data['P_MIN'], data['NUM_SOURCE'], data['NUM_DEMAND'])
    grid_out = maxmin_normal.decodenew(u)
    # 已知输入边界条件
    in_BC = maxmin_normal.decodebatch(batch[0])
    BC_G = in_BC[:, :, 0, -data['NUM_DEMAND'] - data['NUM_SOURCE']:-data['NUM_SOURCE']].permute(0, 2, 1)
    BC_P = in_BC[:, :, 0, -data['NUM_SOURCE']:].permute(0, 2, 1)

    # 初始化损失字典，为每条管道计算loss
    inject_nodes = list(set(data['gas_net'].nodes) - set(data['demand_nodes']) - set(data['source_nodes']))  # 汇节点
    loss = {key: torch.zeros(data['NUM_PIPES'], device=u[0].device) for key in ['IC1', 'IC2', 'EQ1', 'EQ2']}
    loss.update({key: torch.zeros(1, device=u[0].device) for key in ['BC_G1', 'BC_G2', 'BC_P1', 'BC_P2']})
    # loss['NODE'] = torch.zeros(len(inject_nodes), device=u[0].device)
    S_loss = {key: torch.zeros(data['NUM_PIPES'], device=u[0].device) for key in
              ['T_Flow', 'T_Pres', 'R_Flow', 'R_Pres', 'Flow', 'Pressure']}
    R_L2_G_P = [[], []]
    for i, key in enumerate(list(data['gas_net'].edges())):
        pipe_label = data['pipe_lable'][key]
        l, D, Z, R, Temp, frac = data['PIPES_LEN'][pipe_label], data['DIAMETER'][pipe_label], data['Z_FACTOR'][pipe_label], \
            data['R_FACTOR'][pipe_label], data['T_FACTOR'][pipe_label], data['FRICTION'][pipe_label]
        area = np.pi * D ** 2 / 4
        dx, dt = torch.tensor(data['SUB_X'], device=u[0].device), torch.tensor(data['SUB_T'], device=u[0].device)
        uG, uP = grid_out[i][..., 0], grid_out[i][..., 1]
        Pv, Pou = uG / area, uP / (Z * R * Temp)
        V = Pv / Pou

        # 计算IC loss
        # 方程1  由Pou_t = 0，稳定流动
        loss['IC1'][i] += rmse_loss((Pv[:, 0, 1:] - Pv[:, 0, :-1]) / dx * dt * (Z * R * Temp), torch.zeros(Pv[:, 0, 1:].shape, device=u[0].device))
        # 方程2 由pv_t=0 #离散形式，有误差
        P_x = (uP[:, 0, 1:] - uP[:, 0, :-1]) / dx
        fPouvv = frac * Pv[:, 0] * abs(V[:, 0]) / (2 * D)
        fPouvv = (fPouvv[:, :-1] + fPouvv[:, 1:]) / 2
        loss['IC2'][i] += rmse_loss((P_x + fPouvv) * dx, torch.zeros(P_x.shape, device=u[0].device))

        # 计算方程loss
        en = convolution(Pou, 't') / dt + convolution(Pv, 'x') / dx
        NS = convolution(Pv, 't') / dt + convolution(uP, 'x') / dx + frac * convolution(Pv * abs(V), 'ones') / (2 * D)
        loss['EQ1'][i] += rmse_loss(en * dt * (Z * R * Temp), torch.zeros(en.shape, device=u[0].device))  # e-6
        loss['EQ2'][i] += rmse_loss(NS * dx, torch.zeros(NS.shape, device=u[0].device))  # e2

        # 计算仿真数据损失
        if labels is not None:
            uG2 = labels['Flow'][i]
            uP2 = labels['Pressure'][i]
            S_loss['Flow'][i] += rmse_loss(uG, uG2) * (Z * R * Temp)
            S_loss['Pressure'][i] += rmse_loss(uP, uP2)
            S_loss['R_Flow'][i] += torch.mean(relative_error(uG, uG2))
            S_loss['R_Pres'][i] += torch.mean(relative_error(uP, uP2))
            S_loss['T_Flow'][i] += F.l1_loss(uG, uG2)
            S_loss['T_Pres'][i] += F.l1_loss(uP, uP2)
            R_L2_G_P[0].append(relative_error(uG, uG2))
            R_L2_G_P[1].append(relative_error(uP, uP2))

    # 计算BC loss
    for node in list(data['gas_net'].nodes()):
        if node in data['demand_nodes']:
            connected_edges = list(data['gas_net'].edges(node))
            pipe = [data['pipe_lable'][edge] for edge in connected_edges]
            loss['BC_G1'] += rmse_loss(grid_out[pipe[0]][:, :, -1, 0], BC_G[:, data['demand_nodes'].index(node)]) * (Z * R * Temp)
        elif node in data['source_nodes']:
            connected_edges = list(data['gas_net'].edges(node))
            pipe = [data['pipe_lable'][edge] for edge in connected_edges]
            loss['BC_P1'] += rmse_loss(grid_out[pipe[0]][:, :, 0, 1], BC_P[:, data['source_nodes'].index(node)])
        elif node in inject_nodes:
            inpipes = [('A', 'B')]
            outpipes = [('B', 'C'), ('B', 'D')]

            g_in = sum(grid_out[data['pipe_lable'][edge]][:, :, -1, 0] for edge in inpipes)
            g_out = sum(grid_out[data['pipe_lable'][edge]][:, :, 0, 0] for edge in outpipes)
            loss['BC_G2'] += rmse_loss(g_in, g_out) * (Z * R * Temp)  # 多除这个2是因为有两条输出管道

            p_cross = torch.stack([grid_out[data['pipe_lable'][edge]][:, :, -1, 1] for edge in inpipes] + [grid_out[data['pipe_lable'][edge]][:, :, 0, 1] for edge in outpipes])
            BC_P_insection = sum(rmse_loss(value1, value2) for pp, value1 in enumerate(p_cross) for value2 in p_cross[pp + 1:])
            loss['BC_P2'] += BC_P_insection / (len(p_cross) * (len(p_cross) - 1) / 2)

    # 各项权重
    EQ_weight = config['train']['EQ_loss']
    BC_weight = config['train']['BC_loss']
    IC_weight = config['train']['IC_loss']

    mean_loss = {}
    mean_loss['IC'] = IC_weight * (torch.sum(loss['IC1']) * 2.5 + torch.sum(loss['IC2']) * 30) / data['NUM_PIPES']
    mean_loss['EQ'] = EQ_weight * (torch.sum(loss['EQ1']) * 2.5 + torch.sum(loss['EQ2']) * 30) / data['NUM_PIPES']
    mean_loss['BC_G'] = BC_weight * (loss['BC_G1'] / len(data['demand_nodes']) + loss['BC_G2'] / len(inject_nodes))
    mean_loss['BC_P'] = BC_weight * (loss['BC_P1'] / len(data['source_nodes']) + loss['BC_P2'] / len(inject_nodes))

    sum_loss = sum(mean_loss.values())
    for key in ['Flow', 'Pressure', 'R_Flow', 'R_Pres', 'T_Flow', 'T_Pres']:
        S_loss[key] = torch.sum(S_loss[key]) / data['NUM_PIPES']
    if labels is not None:
        sum_loss = (S_loss['Flow'] + S_loss['Pressure']) / 2

    return sum_loss, mean_loss, S_loss, R_L2_G_P


def train(method, input_GRF, input_sim, out_simulation, optimizer_YN='milestones', save_YN=False):  # optimizer=milestones,lr
    data_dict1 = torch.load(input_GRF)  # 加载采样数据
    data_dict2 = torch.load(input_sim)  # 加载仿真数据
    simulations = load_simulation_data(config['train']['Nsample'], out_simulation)  # 加载仿真结果数据

    # 使用自定义的 Dataset 和 BatchSampler
    combined_dataset = CombinedDataset([sample[:config['train']['Nsample']] for sample in data_dict1['train_sample']],
                                       [sample[:config['train']['Nsample']] for sample in data_dict2['train_sample']], labels=simulations)
    batch_sampler = CustomBatchSampler(combined_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset=combined_dataset, batch_sampler=batch_sampler, collate_fn=simple_collate_fn)
    # 设置优化器和学习率调度器
    if optimizer_YN == 'lr':
        optimizer = Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['scheduler_step'],
                                                    gamma=config['train']['scheduler_gamma'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80,
        #                                                        threshold=1e-4)

    else:  # optimizer_YN == 'milestones':
        optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['scheduler_gamma'])
    # 初始化 last_lr 为当前学习率
    last_lr = scheduler.get_last_lr()

    # 加载检查点（如果有）
    checkpoint_path = "None"
    # checkpoint_path = 'checkpoint_s/3Y_gas/v61+2024-10-13_01-24-19+Epochs_500+Mode_t_25+Modes+x_25_32/32_v61+Epoch_300+Nsample_1000+Modes1_25+Modes2_25+Loss_681.6216.pt'
    start_epoch, current_time = load_checkpoint(model, optimizer, scheduler, checkpoint_path=checkpoint_path)
    setup_logging(current_time)

    model.train()
    t3 = default_timer()
    epochs = config['train']['epochs'][0] if optimizer_YN == 'lr' else config['train']['epochs'][1]
    for ep in range(start_epoch, epochs):
        t1 = default_timer()
        total_loss = 0
        # 直接使用 batch_sampler 进行迭代处理
        for batch_data, batch_labels in dataloader:
            batch_data = [tensor.to(device) for tensor in batch_data]
            if batch_labels is not None:
                batch_labels = {key: [tensor.to(device) for tensor in tensors] for key, tensors in batch_labels.items()}

            # 前向传播
            optimizer.zero_grad()
            out = model(batch_data, method)

            # 计算损失
            batch_loss, mean_loss, S_loss, _ = PINO_loss(out, batch_data, batch_labels, data_dict1=data_dict1, data_dict2=data_dict2)

            # 反向传播和优化
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()

        # 当学习率发生变化时输出
        if current_lr != last_lr:
            print(f"Learning rate changed to: {current_lr}")
            last_lr = current_lr
        t2 = default_timer()

        total_loss /= len(dataloader)

        modes_t, modes_x = config['model']['modes1'][0], config['model']['modes2'][0]
        current_file = os.path.basename(__file__)[4:7]
        path1 = f'{current_file}+{current_time}+Epochs_{epochs}+Mode_t_{modes_t}+Modes+x_{modes_x}'
        save_name = f'{method}_{current_file}+Epoch_{ep}+Nsample_{len(combined_dataset)}+Modes1_{modes_t}+Modes2_{modes_x}+Loss_{total_loss:.4f}'
        # 保存中间训练模型
        if ep % (epochs * 0.2) == 0 and ep != 0:
            save_checkpoint(config['train']['save_dir'], path1, save_name + '.pt', model, ep, total_loss, optimizer, scheduler)
        print(
            f'ep: {ep}/{epochs}, t: {(t2 - t1):.3f}, train_loss: {total_loss:.4f}, '
            f'batch_loss: {batch_loss.item():.4f}, IC: {mean_loss["IC"]:.4f}, '
            f'EQ: {mean_loss["EQ"]:.4f}, BC_G: {mean_loss["BC_G"].item():.4f}, '
            f'BC_P: {mean_loss["BC_P"].item():.4f}, '
            f'T_f: {S_loss["T_Flow"]:.4f}, T_Pres: {S_loss["T_Pres"]:.4f}'
        )
        logging.info(
            f'ep: {ep}/{epochs}, t: {(t2 - t1):.3f}, train_loss: {total_loss:.4f}, '
            f'batch_loss: {batch_loss.item():.4f}, IC: {mean_loss["IC"]:.4f}, '
            f'EQ: {mean_loss["EQ"]:.4f}, BC_G: {mean_loss["BC_G"].item():.4f}, '
            f'BC_P: {mean_loss["BC_P"].item():.4f}, '
            f'T_f: {S_loss["T_Flow"]:.4f}, T_Pres: {S_loss["T_Pres"]:.4f}'
        )
    # 保存训练模型
    t4 = default_timer()
    if save_YN == True:
        save_checkpoint(config['train']['save_dir'], path1, save_name + '.pt', model, ep, total_loss, optimizer, scheduler)

    print(f'train done! t:{(t4 - t3):.3f}')
    print(save_name)


def test(input_test, modelPATH, out_test_sim):
    # #加载模型参数
    print('\n', modelPATH)
    data_dict3 = torch.load(input_test)  # 加载测试数据
    checkpoint = torch.load(modelPATH)  # 加载 checkpoint
    model.load_state_dict(checkpoint['model'])  # 加载模型权重
    # 不同分辨率
    test_name = [[640, 400], [320, 200], [160, 100]]
    # test_name = [[160, 100]]

    num_sample = 200
    for subt, subx in test_name:
        print(f'subt:{subt}, subx:{subx}')
        test_sim = load_simulation_data(num_sample, out_test_sim, subt, subx)  # 加载仿真结果数据
        data_dict3['SUB_X'] = subx
        data_dict3['SUB_T'] = subt
        test_data = []
        for i in range(data_dict3['NUM_PIPES']):
            test_data.append(data_dict3['train_sample'][i][:, ::subt // 80, ::subx // 50, :])
        combined_dataset = CombinedDataset(None, [sample[:num_sample] for sample in test_data], labels=test_sim)
        batch_sampler = CustomBatchSampler(combined_dataset, batch_size=config['train']['batchsize'], shuffle=False)
        dataloader = torch.utils.data.DataLoader(dataset=combined_dataset, batch_sampler=batch_sampler, collate_fn=simple_collate_fn)

        model.eval()
        total_loss = {}
        total_loss[f'subt_{subt}+subx_{subx}'] = 0.0
        total_R_l2 = [[], []]
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = [tensor.to(device) for tensor in batch_data]
                batch_labels = {key: [tensor.to(device) for tensor in tensors] for key, tensors in batch_labels.items()}

                out = model(batch_data, method)
                batch_loss, mean_loss, S_loss, batch_R_L2_G_P = PINO_loss(out, batch_data, batch_labels, data_dict3=data_dict3)
                total_R_l2[0].append(batch_R_L2_G_P[0])
                total_R_l2[1].append(batch_R_L2_G_P[1])
                del out
                torch.cuda.empty_cache()
                total_loss[f'subt_{subt}+subx_{subx}'] += batch_loss.item()
            all_tensors_G = torch.cat([torch.flatten(t) for sublist in total_R_l2[0] for t in sublist])
            all_tensors_P = torch.cat([torch.flatten(t) for sublist in total_R_l2[1] for t in sublist])
            print(f'mean_G: {torch.mean(all_tensors_G):.4f}, var_G: {torch.std(all_tensors_G):.4f},mean_P: {torch.mean(all_tensors_P):.4f}, var_G: {torch.std(all_tensors_P):.4f}')

            total_loss[f'subt_{subt}+subx_{subx}'] /= len(batch_sampler)
            print(
                f"train_loss: {total_loss[f'subt_{subt}+subx_{subx}']:.4f}, "
                f'batch_loss: {batch_loss.item():.4f}, IC: {mean_loss["IC"]:.4f}, '
                f'EQ: {mean_loss["EQ"]:.4f}, BC_G: {mean_loss["BC_G"].item():.4f}, '
                f'BC_P: {mean_loss["BC_P"].item():.4f}, '
                f'T_f: {S_loss["T_Flow"]:.4f}, T_Pres: {S_loss["T_Pres"]:.4f}'
            )



###########################################################################################
######  开始   ####################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config_file = 'configs/pretrain/network-pretrain_v61.yaml'
config = load_config(config_file)  # 加载配置文件
method = config['model']['method']
model = GNO2d(in_dim=config['model']['in_dim'],
              out_dim=config['model']['out_dim'],
              modes1=config['model']['modes1'],
              modes2=config['model']['modes2'],
              fc_dim=config['model']['fc_dim'],
              layers=config['model']['layers'],
              act=config['model']['L_F_act'],
              pad_ratio=config['model']["pad_ratio"]).to(device)
################################################################################################
if __name__ == "__main__":
    input_GRF = 'data/certain/3Ynet_square_Nsample_500+subt_320s+subx_200m_+Tinter_24.0h.pt'
    input_sim = 'data/certain/3Ynet_square_Nsample_1000+subt_640s+subx_400m_+Tinter_24.0h.pt'
    out_simulation = 'pipelinestudio/result_npy/Nsample_1000/640s_400m/pipe3_1000N_640s_400m_{:s}.npy'
    # train(method, input_GRF, input_sim, out_simulation, optimizer_YN='lr', save_YN=True)  # milestones,lr

    ############# Test ###############################################################################
    input_test = 'data/certain/3Ynet_square_Nsample_200+subt_80s+subx_50m_+Tinter_24.0h.pt'
    modelPATH = 'checkpoint_s/3Y_gas/v61+2024-10-19_10-42-55+Epochs_500+Mode_t_25+Modes+x_25_39_pad=[0,0.001,0]/39_v61+Epoch_300+Nsample_1000+Modes1_25+Modes2_25+Loss_224.9358.pt'
    out_test_sim = 'pipelinestudio/result_npy/Nsample_200/{:d}s_{:d}m/pipe3_200N_{:d}s_{:d}m_{:s}.npy'
    test(input_test, modelPATH, out_test_sim)
