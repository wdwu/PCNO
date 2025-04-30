# from utilities3 import GaussianRF, save_checkpoint
import os
import re
import logging
import random
import yaml
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from train_utils.adam import Adam
from torchinfo import summary
#############################################################
# V61版本 融合仿真数据和采样数据训练。
#############################################################
def setup_logging(current_time, path, log_prefix='training'):
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = './results/log/%s/' % (path)
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
    if max(pad_ratio) <= 0:
        return x, [0, 0], [[0, 0]] * len(x), [0, 0]

    GRID_T = x[0].size(2)
    NUM_PIPES = len(x)
    res = x.copy()

    # 时间维度补零
    num_pad1 = [round(GRID_T * pad_ratio[0])] * 2

    # 空间维度补零
    max_length = max(GRID_X)
    pad2 = [round(max_length * pad_ratio[1])] * 2
    num_pad2 = []
    for i in range(NUM_PIPES):
        pad_left = pad2[0] + round((max_length - GRID_X[i]) / 2)
        pad_right = sum(pad2) + max_length - GRID_X[i] - pad_left
        num_pad2.append([pad_left, pad_right])
        res[i] = F.pad(x[i], (num_pad2[i][0], num_pad2[i][1], num_pad1[0], num_pad1[1]), 'constant', 0)

    # 管道维度补零
    num_pad3 = [round(NUM_PIPES * pad_ratio[2])] * 2
    if max(num_pad3) > 0:
        res = torch.stack(res, dim=-1)
        res = F.pad(res, (num_pad3[0], num_pad3[1]))
        res = list(torch.unbind(res, dim=-1))

    return res, num_pad1, num_pad2, num_pad3

def remove_padding2(x, num_pad1, num_pad2, num_pad3):
    # 移除管道维度的填充
    if num_pad3[0] > 0 or num_pad3[1] > 0:
        x = x[num_pad3[0]:len(x) - num_pad3[1]]

    # 移除时间和空间维度的填充
    results = []
    for tensor, (pad_left, pad_right) in zip(x, num_pad2):
        if num_pad1[0] > 0 or num_pad1[1] > 0 or pad_left > 0 or pad_right > 0:
            h_start = num_pad1[0]
            h_end = tensor.size(2) - num_pad1[1]
            w_start = pad_left
            w_end = tensor.size(3) - pad_right
            tensor = tensor[:, :, h_start:h_end, w_start:w_end]
        results.append(tensor)

    # 合并所有管道并调整维度顺序
    return torch.cat(results, dim=-1).permute(0, 2, 3, 1)

# 保存检查点
def save_checkpoint(path, path1, name, model, ep, sum_loss, optimizer=None, scheduler=None):
    # 构建保存目录路径
    ckpt_dir = os.path.join('./results/checkpoint_s', path, path1)

    # 确保目录存在
    os.makedirs(ckpt_dir, exist_ok=True)

    # 获取模型状态字典
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    # 构建保存字典
    save_dict = {
        'model': model_state_dict,
        'ep': ep,
        'sum_loss': sum_loss,
    }

    # 可选保存优化器和调度器状态
    if optimizer is not None:
        save_dict['optim'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()


    # 构建完整文件路径并保存
    save_path = os.path.join(ckpt_dir, name)
    torch.save(save_dict, save_path)
    # print('Checkpoint is saved at %s' % ckpt_dir + name)

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

class MaxMinNormal:
    def __init__(self,G_MAX, G_MIN, P_MAX, P_MIN, NUM_SOURCE, NUM_DEMAND):
        # 流量参数
        self.G_range = (G_MIN, G_MAX)
        self.G_scale = G_MAX - G_MIN
        self.G_offset = G_MIN
        # 压力参数
        self.P_range = (P_MIN, P_MAX)
        self.P_scale = P_MAX - P_MIN
        self.P_offset = P_MIN
        # 节点配置
        self.source_slice = slice(-NUM_SOURCE, None)
        self.demand_slice = slice(-NUM_SOURCE - NUM_DEMAND, -NUM_SOURCE)
        # 预计算常用值
        self._scales = torch.tensor([self.G_scale, self.P_scale])
        self._offsets = torch.tensor([self.G_offset, self.P_offset])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """选择性标准化特定通道"""
        x_norm = x.clone()
        x_norm[..., self.demand_slice] = (x[..., self.demand_slice] - self.G_offset) / self.G_scale
        x_norm[..., self.source_slice] = (x[..., self.source_slice] - self.P_offset) / self.P_scale
        return x_norm

    def decode(self, x_list):
        """反标准化张量列表"""
        original_sizes = [t.size(2) for t in x_list]
        x_cat = torch.cat(x_list, dim=2)
        x_cat[..., 0] = x_cat[..., 0] * self.G_scale + self.G_offset  # 流量
        x_cat[..., 1] = x_cat[..., 1] * self.P_scale + self.P_offset  # 压力
        return list(torch.split(x_cat, original_sizes, dim=2))

    def decodebatch(self, x: torch.Tensor) -> torch.Tensor:
        """选择性反标准化"""
        x_denorm = x.clone()
        x_denorm[..., self.demand_slice] = x[..., self.demand_slice] * self.G_scale + self.G_offset
        x_denorm[..., self.source_slice] = x[..., self.source_slice] * self.P_scale + self.P_offset
        return x_denorm

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
                            for j in range(config['model']['num_pipe'])]
    return simulations

def relative_error(y_pred, y_true, mod, epsilon=1e-16):
    if mod=='global':
        num_examples = y_pred.shape[0]
        numerator = torch.norm(y_true.reshape(num_examples,-1)  - y_pred.reshape(num_examples,-1) , p=2 ,dim=1)
        denominator = torch.norm(y_true.reshape(num_examples,-1) , p=2)
        relative_error = torch.mean(numerator / (denominator + epsilon))
    elif mod=='point':
        absolute_error = torch.abs(y_pred - y_true)
        relative_error = absolute_error / (torch.abs(y_true) + epsilon)
    return relative_error

def rmse_loss(y_pred, y, epsilon=1e-16):
    mse = F.mse_loss(y_pred, y)
    return torch.sqrt(mse + epsilon)


class Graph_SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(Graph_SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = config['model']['num_pipe']
        self.num_pipe = config['model']['num_pipe']

        # self.w_conv2d = nn.Conv2d(in_channels, out_channels, 1)
        # self.w_conv2d2 = nn.Conv2d(in_channels, out_channels, 1)
        self.act = _get_act('relu', 'complex')
        self.act_w = _get_act('relu', 'real')
        self.scale = (1 / (in_channels * out_channels))
        # 根据方法初始化权重
        self._init_weights(method)

    def _init_weights(self, method):

        if method in [26]:
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights3 = nn.Parameter(
                self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(
                self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        if method in [34]:
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        if method in [25]:
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, self.modes3, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, self.modes3, self.modes1, self.modes2, dtype=torch.cfloat))
        if method in [27]:
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels,  self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels,  self.modes1, self.modes2, dtype=torch.cfloat))

        if method in [35,39]:
            self.weights10 = nn.Parameter(self.scale * torch.rand(self.num_pipe, self.num_pipe, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights11 = nn.Parameter(self.scale * torch.rand(self.num_pipe, self.num_pipe, self.modes1, self.modes2, dtype=torch.cfloat))
        if method in [35]:
            self.weights12 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, dtype=torch.cfloat))
            self.weights13 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, dtype=torch.cfloat))
        if method in [8, 38, 39]:
            self.weights15 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, self.num_pipe, dtype=torch.cfloat))
            self.weights16 = nn.Parameter(self.scale * torch.rand(self.out_channels, self.out_channels, self.num_pipe, dtype=torch.cfloat))

    def forward(self, x, method, num_pad1, num_pad2, num_pad3, kernel_low=None, kernel_high=None):
        batchsize = x[0].size(0)

        # 公共预处理：傅里叶变换和模式提取
        def preprocess(x_list):
            fno_in, fno_in_h = [], []
            for tensor in x_list:
                x_ft = torch.fft.rfftn(tensor, dim=[2, 3])
                if method in [1, 2, 3, 6, 8, 12, 13, 15, 19, 20, 22, 23, 24, 25, 27, 32, 35, 39]:
                    fno_in.append(x_ft[:, :, :self.modes1, :self.modes2].unsqueeze(2))
                    fno_in_h.append(x_ft[:, :, -self.modes1:, :self.modes2].unsqueeze(2))
            return torch.cat(fno_in, dim=2), torch.cat(fno_in_h, dim=2)

        # 公共后处理：逆变换重构
        def postprocess(x_ref, f_out, f_out_h):
            outputs = []
            for i in range(self.num_pipe):
                out_tensor = torch.zeros(batchsize, self.out_channels,
                                         x_ref[i].size(2), x_ref[i].size(3),
                                         device=x_ref[i].device, dtype=torch.cfloat)
                out_tensor[:, :, :self.modes1, :self.modes2] = f_out[:, :, i]
                out_tensor[:, :, -self.modes1:, :self.modes2] = f_out_h[:, :, i]
                outputs.append(torch.fft.irfftn(out_tensor, s=(x_ref[i].size(2), x_ref[i].size(3)), dim=[2, 3]))
            return outputs

        # 方法分发
        if method == 25:#FNO-A
            fno_in, fno_in_h = preprocess(x)
            fno_out = torch.einsum("bizxy,iozxy->bozxy", fno_in, self.weights1)
            fno_out_h = torch.einsum("bizxy,iozxy->bozxy", fno_in_h, self.weights4)
            return postprocess(x, fno_out, fno_out_h)

        elif method == 27: #FNO
            fno_in, fno_in_h = preprocess(x)
            fno_out = torch.einsum("bizxy,ioxy->bozxy", fno_in, self.weights1)
            fno_out_h = torch.einsum("bizxy,ioxy->bozxy", fno_in_h, self.weights4)
            return postprocess(x, fno_out, fno_out_h)
        # FNO-3D
        elif method == 26:
            x_stack = torch.stack(x, dim=-1)
            x_ft = torch.fft.rfftn(x_stack, dim=[2, 3, 4])
            z_dim = min(x_ft.shape[4], self.modes3)

            out = torch.zeros(batchsize, self.out_channels,
                              x_stack.size(2), x_stack.size(3), x_stack.size(4),
                              device=x_stack.device, dtype=torch.cfloat)

            quadrants = [
                (slice(None), slice(None), slice(None, self.modes1), slice(None, self.modes2), self.weights1),
                (slice(None), slice(None), slice(-self.modes1, None), slice(None, self.modes2), self.weights2),
                (slice(None), slice(None), slice(None, self.modes1), slice(-self.modes2, None), self.weights3),
                (slice(None), slice(None), slice(-self.modes1, None), slice(-self.modes2, None), self.weights4)
            ]

            for q in quadrants:
                coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3,
                                    device=x_stack.device, dtype=torch.cfloat)
                coeff[..., :z_dim] = x_ft[q[0], q[1], q[2], q[3], :z_dim]
                out[q[0], q[1], q[2], q[3], :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", coeff, q[4])

            return list(torch.unbind(
                torch.fft.irfftn(out, s=(x_stack.size(2), x_stack.size(3), x_stack.size(4)), dim=[2, 3, 4]),
                dim=-1))

        # PCNO-3D
        elif method == 34:
            x_stack = torch.stack(x, dim=-1)
            x_ft = torch.fft.rfftn(x_stack, dim=[2, 3, 4])
            z_dim = min(x_ft.shape[4], self.modes3)

            out = torch.zeros_like(x_stack, dtype=torch.cfloat)

            # 两象限处理
            for quadrant, weight in [
                ((slice(None), slice(None), slice(None, self.modes1), slice(None, self.modes2)), self.weights1),
                ((slice(None), slice(None), slice(-self.modes1, None), slice(None, self.modes2)), self.weights2)
            ]:
                coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3,
                                    device=x_stack.device, dtype=torch.cfloat)
                coeff[..., :z_dim] = x_ft[quadrant[0], quadrant[1], quadrant[2], quadrant[3], :z_dim]
                out[quadrant[0], quadrant[1], quadrant[2], quadrant[3], :self.modes3] = \
                    torch.einsum("bixyz,ioxyz->boxyz", coeff, weight)

            return list(torch.unbind(
                torch.fft.irfftn(out, s=(x_stack.size(2), x_stack.size(3), x_stack.size(4)), dim=[2, 3, 4]),
                dim=-1))

        # PCNO-C,# PCNO
        elif method in [35, 39]:
            fno_in, fno_in_h = preprocess(x)

            # 公共处理阶段
            fno_out = torch.einsum("bizxy,zoxy->bioxy", fno_in, self.weights10)
            fno_out_h = torch.einsum("bizxy,zoxy->bioxy", fno_in_h, self.weights11)

            # PCNO-C
            if method == 35:
                fno_out = torch.einsum("bizxy,io->bozxy", fno_out, self.weights12)
                fno_out_h = torch.einsum("bizxy,io->bozxy", fno_out_h, self.weights13)
            else:  # method 39 # PCNO
                fno_out = torch.einsum("bizxy,ioz->bozxy", fno_out, self.weights15)
                fno_out_h = torch.einsum("bizxy,ioz->bozxy", fno_out_h, self.weights16)

            return postprocess(x, fno_out, fno_out_h)

        else:
            raise ValueError(f"Unsupported method: {method}")

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
        self.n_pipes = config['model']['num_pipe']  # 管道数固定为3

        # 网络结构
        # 升维
        self.fc0 = nn.Linear(in_dim, layers[0])

        #傅里叶层
        self.graph_convs = nn.ModuleList([
            Graph_SpectralConv2d(in_sz, out_sz, m1, m2)
            for in_sz, out_sz, m1, m2 in zip(layers, layers[1:], modes1, modes2)
        ])

        # 线性层
        self.ws = nn.ModuleList([
            nn.Conv2d(in_sz, out_sz, 1)
            for in_sz, out_sz in zip(layers, layers[1:])
        ])

        # 全连接层
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act, 'real')

    def forward(self, x, method, kernel_low=None, kernel_high=None):
        # 输入预处理
        original_sizes = [t.size(2) for t in x]
        x_cat = torch.cat(x, dim=2)
        # n_pipes = len(x)

        # 初始全连接
        x = self.fc0(x_cat).permute(0, 3, 1, 2)
        x_split = list(torch.split(x, original_sizes, dim=3))

        # 填充处理
        x_padded, pad1, pad2, pad3 = add_padding2(x_split, self.pad_ratio, original_sizes)

        # 卷积处理
        for i, (spec_conv, w_conv) in enumerate(zip(self.graph_convs, self.ws)):
            # 分支处理
            if kernel_low is None:
                x_spec = spec_conv(x_padded, method, pad1, pad2, pad3)
                x_padded = [x_spec[j] + w_conv(x_padded[j]) for j in range(self.n_pipes)]
            else:
                x_padded = spec_conv(x_padded, method, self.pad_ratio, kernel_low[i], kernel_high[i])

            # 激活函数（最后一层不加）
            if i != len(self.graph_convs) - 1:
                x_padded = [self.act(t) for t in x_padded]

        # 移除填充
        x = remove_padding2(x_padded, pad1, pad2, pad3)

        # 全连接处理
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        # x = self.act(x)  # 另外加的，加了后loss降得慢

        # 输出验证和分割
        if x.shape[-1] != self.out_dim:
            raise ValueError(f'Output dimension mismatch: got {x.shape[-1]}, expected {self.out_dim}')

        x = list(torch.split(x, original_sizes, dim=2))
        return x

#####train############################################################
def load_checkpoint(model, optimizer, scheduler, checkpoint_path="None"):
    # 默认时间戳
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isfile(checkpoint_path):
        return 0, current_time

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None and 'optim' in checkpoint:
            optimizer.load_state_dict(checkpoint['optim'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        time_match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', checkpoint_path)
        current_time = time_match.group() if time_match else current_time
        return checkpoint.get('ep', 0) + 1, current_time

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")


def convolution(image, difference, theta=0.8):
    # 核权重定义
    kernel_dict = {
        'x': [[-(1-theta), (1-theta)], [-theta, theta]],
        't': [[-0.5, -0.5], [0.5, 0.5]],
        'ones': [[(1-theta)/2, (1-theta)/2], [theta/2, theta/2]]
    }
    if difference not in kernel_dict:
        raise ValueError(f"Unsupported convolution type: {difference}. Use 'x', 't' or 'ones'")

    # 准备卷积核
    weights = torch.tensor(kernel_dict[difference], dtype=torch.float32, device=image.device)

    # 构建卷积层 (无梯度计算)
    with torch.no_grad():
        conv = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
        conv.weight = nn.Parameter(weights.view(1, 1, 2, 2))
        conv = conv.to(image.device)

    # 统一处理输入维度
    need_squeeze = image.dim() == 2
    input_tensor = image.unsqueeze(0).unsqueeze(0) if need_squeeze else image.unsqueeze(1)

    # 执行卷积
    output = conv(input_tensor)

    # 恢复原始维度
    return output.squeeze(0).squeeze(0) if need_squeeze else output.squeeze(1)

def simple_collate_fn(batch):
    return batch  # 直接返回批次数据，不进行任何处理


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(dataset)

    def __iter__(self):
        # 生成数据集索引
        def gen_indices(length):
            return torch.randperm(length).tolist() if self.shuffle else list(range(length))

        indices1 = gen_indices(self.dataset.dataset1_len)
        indices2 = gen_indices(self.dataset.dataset2_len)

        # 创建批次
        batches = ([(1, indices1[i:i+self.batch_size]) for i in range(0, self.dataset.dataset1_len, self.batch_size)] +
                   [(2, indices2[i:i+self.batch_size]) for i in range(0, self.dataset.dataset2_len, self.batch_size)])

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return (self.dataset.dataset1_len + self.batch_size - 1) // self.batch_size + \
               (self.dataset.dataset2_len + self.batch_size - 1) // self.batch_size


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, labels):
        self.dataset1 = dataset1 or (None,)
        self.dataset2 = dataset2
        self.labels = labels
        self.dataset1_len = len(dataset1[0]) if dataset1 else 0
        self.dataset2_len = len(dataset2[0])

    def __len__(self):
        return self.dataset1_len + self.dataset2_len

    def __getitems__(self, idx):
        dataset_id, indices = idx
        if dataset_id == 1:
            return [tensor[indices] for tensor in self.dataset1], None
        return (
            [tensor[indices] for tensor in self.dataset2],
            {k: [t[indices] for t in v] for k, v in self.labels.items()}
        )


def PINO_loss(uu, x_batch, labels=None, data_dict1=None, data_dict2=None, data_dict3=None):
    # 数据准备
    data = data_dict3 or pull_data(data_dict2 if labels else data_dict1)
    u = uu.copy()
    batch = x_batch.copy()

    # 解码处理
    maxmin_normal = MaxMinNormal(data['G_MAX'], data['G_MIN'], data['P_MAX'], data['P_MIN'], data['NUM_SOURCE'], data['NUM_DEMAND'])
    grid_out = maxmin_normal.decode(u)
    # 已知输入边界条件
    in_BC = maxmin_normal.decodebatch(batch[0])
    BC_G = in_BC[:, :, 0, -data['NUM_DEMAND'] - data['NUM_SOURCE']:-data['NUM_SOURCE']].permute(0, 2, 1)
    BC_P = in_BC[:, :, 0, -data['NUM_SOURCE']:].permute(0, 2, 1)

    # 初始化损失
    device = uu[0].device
    inject_nodes = [n for n in data['gas_net'].nodes if n not in data['demand_nodes'] + data['source_nodes']]
    loss = {k: torch.zeros(data['NUM_PIPES'], device=device) for k in ['IC1','IC2','EQ1','EQ2']}
    loss.update({k: torch.zeros(1, device=device) for k in ['BC_G1','BC_G2','BC_P1','BC_P2']})
    S_loss = {k: torch.zeros(data['NUM_PIPES'], device=device) for k in ['Flow','Pressure','R_Flow','R_Pres','T_Flow','T_Pres']}
    R_L2_G_P = [[], []]


    for i, key in enumerate(list(data['gas_net'].edges())):
        pipe_label = data['pipe_lable'][key]
        l, D, Z, R, Temp, frac = data['PIPES_LEN'][pipe_label], data['DIAMETER'][pipe_label], data['Z_FACTOR'][pipe_label], \
            data['R_FACTOR'][pipe_label], data['T_FACTOR'][pipe_label], data['FRICTION'][pipe_label]
        area = np.pi * D ** 2 / 4
        dx, dt = torch.tensor(data['SUB_X'], device=u[0].device), torch.tensor(data['SUB_T'], device=u[0].device)
        uG, uP = grid_out[i][..., 0], grid_out[i][..., 1]
        Pv, Pou, V = uG / area, uP / (Z * R * Temp), (uG / area) / (uP / (Z * R * Temp))

        # IC loss
        loss['IC1'][i] = rmse_loss((Pv[:,0,1:]-Pv[:,0,:-1])/dx*dt*(Z*R*Temp), torch.zeros_like(Pv[:,0,1:]))
        P_x = (uP[:,0,1:]-uP[:,0,:-1])/dx
        f_term = (frac*Pv[:,0]*abs(V[:,0])/(2*D))[:,:-1] + (frac*Pv[:,0]*abs(V[:,0])/(2*D))[:,1:]
        loss['IC2'][i] = rmse_loss((P_x + f_term/2)*dx, torch.zeros_like(P_x))

        # 方程损失
        en = convolution(Pou,'t')/dt + convolution(Pv,'x')/dx
        NS = convolution(Pv,'t')/dt + convolution(uP,'x')/dx + frac*convolution(Pv*abs(V),'ones')/(2*D)
        loss['EQ1'][i] = rmse_loss(en*dt*(Z*R*Temp), torch.zeros_like(en))
        loss['EQ2'][i] = rmse_loss(NS*dx, torch.zeros_like(NS))

        # 仿真损失
        if labels:
            uG2, uP2 = labels['Flow'][i], labels['Pressure'][i]
            S_loss['Flow'][i] = rmse_loss(uG, uG2) * (Z*R*Temp)
            S_loss['Pressure'][i] = rmse_loss(uP, uP2)
            S_loss['R_Flow'][i] = relative_error(uG, uG2,mod='global')
            S_loss['R_Pres'][i] = relative_error(uP, uP2,mod='global')
            S_loss['T_Flow'][i] += F.l1_loss(uG, uG2)
            S_loss['T_Pres'][i] += F.l1_loss(uP, uP2)
            R_L2_G_P[0].append(relative_error(uG, uG2,mod='point'))
            R_L2_G_P[1].append(relative_error(uP, uP2,mod='point'))

    # 边界条件损失
    for node in data['gas_net'].nodes():
        if node in data['demand_nodes']:
            edges = list(data['gas_net'].in_edges(node))
            pipe = data['pipe_lable'][edges[0]]
            loss['BC_G1'] += rmse_loss(grid_out[pipe][:, :, -1, 0], BC_G[:, data['demand_nodes'].index(node)]) * (Z * R * Temp)
        elif node in data['source_nodes']:
            edges = list(data['gas_net'].out_edges(node))
            pipe = data['pipe_lable'][edges[0]]
            loss['BC_P1'] += rmse_loss(grid_out[pipe][:, :, 0, 1], BC_P[:, data['source_nodes'].index(node)])
        elif node in inject_nodes:

            in_pipes = [data['pipe_lable'][e] for e in data['gas_net'].in_edges(node)]
            out_pipes = [data['pipe_lable'][e] for e in data['gas_net'].out_edges(node)]

            # 流量守恒
            g_in = sum(grid_out[p][:, :, -1, 0] for p in in_pipes)
            g_out = sum(grid_out[p][:, :, 0, 0] for p in out_pipes)
            loss['BC_G2'] += rmse_loss(g_in, g_out) * (Z * R * Temp)

            # 压力连续
            p_values = [grid_out[p][:, :, -1, 1] for p in in_pipes] + [grid_out[p][:, :, 0, 1] for p in out_pipes]
            loss['BC_P2'] += sum(rmse_loss(p_values[i], p_values[j])
                                 for i in range(len(p_values)) for j in range(i + 1, len(p_values))) / (
                                         len(p_values) * (len(p_values) - 1) / 2)

    # 损失加权
    w = config['train']
    mean_loss = {
        'IC': w['IC_loss'] * (loss['IC1'].sum()*2.5 + loss['IC2'].sum()*30)/data['NUM_PIPES'],
        'EQ': w['EQ_loss'] * (loss['EQ1'].sum()*2.5 + loss['EQ2'].sum()*30)/data['NUM_PIPES'],
        'BC_G': w['BC_loss'] * (loss['BC_G1']/len(data['demand_nodes']) + loss['BC_G2']/len(inject_nodes)),
        'BC_P': w['BC_loss'] * (loss['BC_P1']/len(data['source_nodes']) + loss['BC_P2']/len(inject_nodes))
    }

    sum_loss = sum(mean_loss.values())
    for key in ['Flow', 'Pressure', 'R_Flow', 'R_Pres','T_Flow','T_Pres']:
        S_loss[key] = torch.sum(S_loss[key]) / data['NUM_PIPES']
    if labels is not None:
        sum_loss = (S_loss['Flow'] + S_loss['Pressure']) / 2

    return sum_loss, mean_loss, S_loss, R_L2_G_P


def print_memory_usage(model, optimizer=None, scheduler=None):
    """打印模型、优化器和调度器的精确内存占用"""
    # 1. 模型参数内存
    model_params_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    # 2. 优化器状态内存（更精确计算）
    optim_mem = 0.0
    if optimizer is not None:
        for param_group in optimizer.state_dict()['state'].values():
            for k, v in param_group.items():
                if isinstance(v, torch.Tensor):
                    optim_mem += v.numel() * v.element_size()
        optim_mem /= (1024 ** 2)

    # 3. 调度器状态内存
    scheduler_mem = 0.0
    if scheduler is not None:
        for k, v in scheduler.state_dict().items():
            if isinstance(v, torch.Tensor):
                scheduler_mem += v.numel() * v.element_size()
        scheduler_mem /= (1024 ** 2)

    # 4. 总内存
    total_mem = model_params_mem + optim_mem + scheduler_mem

    print(f"模型参数内存: {model_params_mem:.2f}MB")
    if optimizer is not None:
        print(f"优化器状态内存: {optim_mem:.2f}MB  [含动量+二阶矩]")
    if scheduler is not None:
        print(f"调度器状态内存: {scheduler_mem:.2f}MB")
    print(f"理论总内存占用: {total_mem:.2f}MB")
    print(f"实际检查点大小 ≈ {total_mem * 1.1:.2f}MB  (含PyTorch序列化开销)")

def train(method, input_GRF, input_sim, out_simulation, optimizer_YN='milestones', save_YN=False):  # optimizer=milestones,lr
    # 初始化监控
    def get_memory_usage():
        """获取当前进程内存使用(MB)"""
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    def get_gpu_usage():
        """获取GPU内存使用(MB)"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return 0
    # 初始化监控记录
    memory_records = {
        'epoch': [],
        'time': [],
        'cpu_mem': [],
        'gpu_mem': []
    }
    start_time = time.time()

    # 数据加载
    data_dict1 = torch.load(input_GRF, weights_only=False)
    data_dict2 = torch.load(input_sim, weights_only=False)
    simulations = load_simulation_data(config['train']['Nsample'], out_simulation)

    # 数据集准备
    train_samples = [s[:config['train']['Nsample']] for s in data_dict1['train_sample']]
    train_sims = [s[:config['train']['Nsample']] for s in data_dict2['train_sample']]
    dataset = CombinedDataset(train_samples, train_sims, simulations)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=CustomBatchSampler(dataset, config['train']['batchsize'], True),
        collate_fn=simple_collate_fn
    )

    # 优化器设置
    optimizer = Adam(model.parameters(),
                     lr=config['train']['learning_rate'],
                     weight_decay=5e-4 if optimizer_YN == 'lr' else 0)

    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, config['train']['milestones'], config['train']['scheduler_gamma'])
                 if optimizer_YN == 'milestones' else
                 torch.optim.lr_scheduler.StepLR(optimizer, config['train']['scheduler_step'], config['train']['scheduler_gamma']))


    # 加载检查点（如果有）
    checkpoint_path = "None"
    start_epoch, current_time = load_checkpoint(model, optimizer, scheduler, checkpoint_path=checkpoint_path)
    setup_logging(current_time,config['train']['save_dir'])

    # 添加配置文件，模型结构参数量，模型占用内存
    logging.info("YAML 文件内容:\n%s", yaml.dump(config, default_flow_style=False))
    model_stats  = summary(model)
    summary_str = str(model_stats)
    logging.info("\nModel Summary:\n" + summary_str)
    me = {'model_size(MB)': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)}
    print(me)
    logging.info(me)

    # 训练循环
    model.train()
    epochs = config['train']['epochs'][0] if optimizer_YN == 'lr' else config['train']['epochs'][1]
    for ep in range(start_epoch, epochs):
        epoch_start   = time.time()
        loss_accum = {
            "total": 0.0,  # 累积 total_loss
            "R_Flow": 0.0,
            "R_Pres": 0.0,
            "T_Flow": 0.0,
            "T_Pres": 0.0
        }

        for batch_data, batch_labels in dataloader:

            batch_data = [t.to(device) for t in batch_data]
            batch_labels = {k: [t.to(device) for t in v] for k, v in batch_labels.items()} if batch_labels else None

            optimizer.zero_grad()
            out = model(batch_data, method)

            batch_loss, mean_loss, S_loss, _ = PINO_loss(out, batch_data, batch_labels, data_dict1, data_dict2)

            batch_loss.backward()
            optimizer.step()
            # 累积所有损失项
            loss_accum["total"] += batch_loss.item()
            loss_accum["R_Flow"] += S_loss["R_Flow"].item()
            loss_accum["R_Pres"] += S_loss["R_Pres"].item()
            loss_accum["T_Flow"] += S_loss["T_Flow"].item()
            loss_accum["T_Pres"] += S_loss["T_Pres"].item()


        scheduler.step()
        # 计算所有损失的平均值
        num_batches = len(dataloader)
        loss_avg = {k: v / num_batches for k, v in loss_accum.items()}
        # print_memory_usage(model, optimizer, scheduler)
        # 日志记录
        # ===== 新增监控记录 =====
        epoch_time = time.time() - epoch_start
        current_cpu = get_memory_usage()
        current_gpu = get_gpu_usage()

        memory_records['epoch'].append(ep)
        memory_records['time'].append(epoch_time)
        memory_records['cpu_mem'].append(current_cpu)
        memory_records['gpu_mem'].append(current_gpu)

        # 打印监控信息
        log_msg = (f'ep: {ep + 1}/{epochs}, t: {epoch_time:.3f}s, '
                   f'CPU: {current_cpu:.1f}MB, GPU: {current_gpu:.1f}MB | '
                   f'train_loss: {loss_avg["total"]:.4f},'
                   f'RL2_flow: {loss_avg["R_Flow"]:.4f}, RL2_Pres: {loss_avg["R_Pres"]:.4f},'
                   f'T_flow: {loss_avg["T_Flow"]:.4f}, T_Pres: {loss_avg["T_Pres"]:.4f}')

        print(log_msg)
        logging.info(log_msg)

        # 定期保存
        if ep % (epochs//5) == 0 or ep == epochs-1:
            save_name = f'{method}_{os.path.basename(__file__)[4:7]}+{current_time}+Epochs_{epochs}+Nsample_{len(dataset)}'
            save_checkpoint(config['train']['save_dir'],
                          f'{save_name}_Modes{config["model"]["modes1"][0]}_{config["model"]["modes2"][0]}',
                          f'Epoch_{ep}_Loss_{loss_avg["total"]:.4f}.pt',
                          model, ep, loss_avg["total"], optimizer, scheduler)
    total_time = time.time() - start_time
    metrics = {
        'total_time(s)': total_time,
        'avg_epoch_time(s)': total_time/epochs,
        'max_cpu(MB)': max(memory_records['cpu_mem']),
        'max_gpu(MB)': max(memory_records['gpu_mem']) if torch.cuda.is_available() else 0,
        'model_size(MB)': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    }
    logging.info(metrics)




def test(input_test, modelPATH, out_test_sim):
    # #加载模型参数
    print('\n', modelPATH)
    data_dict = torch.load(input_test, weights_only=False)
    if modelPATH:
        checkpoint = torch.load(modelPATH, weights_only=True)
        model.load_state_dict(checkpoint['model'])

    # Test configurations
    test_resolutions = [(640, 400), (320, 200), (160, 100)]#
    num_sample = 200
    results = {}

    for subt, subx in test_resolutions:
        print(f'\nTesting resolution: subt={subt}, subx={subx}')

        # Prepare data
        test_sim = load_simulation_data(num_sample, out_test_sim, subt, subx)
        data_dict.update({'SUB_X': subx, 'SUB_T': subt})

        test_data = [
            data_dict['train_sample'][i][:, ::subt // 160, ::subx // 100, :][:num_sample]
            for i in range(data_dict['NUM_PIPES'])
        ]

        # Create dataset and dataloader
        dataset = CombinedDataset(None, test_data, labels=test_sim)
        batch_sampler = CustomBatchSampler(
            dataset,
            batch_size=config['train']['batchsize'],
            shuffle=False
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=simple_collate_fn
        )

        # Evaluation
        model.eval()
        metrics = {
            'total_loss': 0.0,
            'l2_global_Flow': 0.0,
            'l2_global_Pres': 0.0,
            'R_l2': [[], []],
        }

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = [t.to(device) for t in batch_data]
                batch_labels = {
                    k: [t.to(device) for t in v]
                    for k, v in batch_labels.items()
                }

                # Forward pass and loss calculation
                out = model(batch_data, method)
                batch_loss, mean_loss, S_loss, batch_R_L2 = PINO_loss(
                    out, batch_data, batch_labels, data_dict3=data_dict
                )

                # Update metrics
                metrics['total_loss'] += batch_loss.item()
                metrics['l2_global_Flow'] += S_loss['R_Flow'].item()
                metrics['l2_global_Pres'] += S_loss['R_Pres'].item()

                # Clean up
                del out
                torch.cuda.empty_cache()

            # Calculate and print results
            metrics['total_loss'] /= len(batch_sampler)
            metrics['l2_global_Flow'] /= len(batch_sampler)
            metrics['l2_global_Pres'] /= len(batch_sampler)


            """Print formatted test results"""
            log_msg = (f"\nResolution subt={subt}, subx={subx} results:"
                       f"\nFlow Statistics(L2_global) - {metrics['l2_global_Flow']:.4f}"
                       f"\nPressure Statistics(L2_global) - {metrics['l2_global_Pres']:.4f}"
                       )

            print(log_msg)
            logging.info(log_msg)






###########################################################################################
######  开始   ####################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config_file = 'configs/pretrain/net_v61_compare.yaml'
with open(config_file) as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)# 加载配置文件
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
    if config['train']['mode'] == 'train':
        input_GRF = './data/certain/3Ynet_square_Nsample_500+subt_320s+subx_200m_+Tinter_24.0h.pt'
        input_sim = './data/certain/3Ynet_square_Nsample_1000+subt_640s+subx_400m_+Tinter_24.0h.pt'
        out_simulation = './data/pipelinestudio/result_npy/3Y_N1000_640S_400M/pipe3_1000N_640s_400m_{:s}.npy'
        input_test = './data/certain/3Ynet_square_Nsample_200+subt_160s+subx_100m_+Tinter_24.0h.pt'
        out_test_sim = './data/pipelinestudio/result_npy/3Y_N200_{:d}S_{:d}M/pipe3_200N_{:d}s_{:d}m_{:s}.npy'

        # input_GRF = './data/certain/9Ynet_square_Nsample_500+subt_320s+subx_200m_+Tinter_24.0h.pt'
        # input_sim = './data/certain/9Ynet_square_Nsample_500+subt_640s+subx_400m_+Tinter_24.0h.pt'
        # out_simulation = './data/pipelinestudio/result_npy/9Y_N500_640S_400M/pipe9_500N_640s_400m_{:s}.npy'
        # input_test = './data/certain/9Ynet_square_Nsample_200+subt_160s+subx_100m_+Tinter_24.0h.pt'
        # out_test_sim = './data/pipelinestudio/result_npy/9Y_N200_{:d}S_{:d}M/pipe9_200N_{:d}s_{:d}m_{:s}.npy'

        train(method, input_GRF, input_sim, out_simulation, optimizer_YN='lr', save_YN=True) # milestones,lr
        test(input_test, None, out_test_sim)

    else:
    ############# Test ###############################################################################

        input_test = './data/certain/3Ynet_square_Nsample_200+subt_160s+subx_100m_+Tinter_24.0h.pt'
        modelPATH = 'results/checkpoint_s/3Y_gas/39_v61+2025-04-25_20-43-55+Epochs_300+Nsample_1000_Modes25_25_pad[0,0.001,0]/Epoch_299_Loss_228.3772.pt'
        out_test_sim = './data/pipelinestudio/result_npy/3Y_N200_{:d}S_{:d}M/pipe3_200N_{:d}s_{:d}m_{:s}.npy'
        test(input_test, modelPATH, out_test_sim)
