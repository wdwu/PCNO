import numpy as np
import scipy.io

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
import math
from torch.utils.data import Dataset
from .utils import get_grid3d, convert_ic, torch2dgrid
from train_utils.random_fields import GaussianRF
from train_utils.utils import maxminnormal


def online_loader(sampler, T, time_scale, modes, G_max, G_min, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize * 50)
        s_max = u0.max()
        s_min = u0.min()
        GRF_data = (u0 - s_min / (s_max - s_min)) * (G_max - G_min) + G_min
        a = convert_ic(GRF_data, batchsize, T, time_scale=time_scale)

        dataset = torch.utils.data.TensorDataset(a)
        if modes == 'train':
            loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)
        return loader
        # yield a


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class DataProcess(object):
    def __init__(self, dataconfig, new=False):
        # dataloader = MatReader(datapath) #以后用来读取真实数据
        self.edge_index = np.array(dataconfig['edge_index'])
        self.num_nodes = np.array(dataconfig['num_nodes'])
        self.num_pipes = np.array(dataconfig['num_pipes'])
        self.lenth = np.array(dataconfig['f_lenth'])
        self.time_interval = np.array(dataconfig['time_interval'])
        self.n_sample = np.array(dataconfig['BC_sample'])
        self.sub_x = np.array(dataconfig['sub_x'])
        self.sub_t = np.array(dataconfig['sub_t'])
        self.X = (self.lenth // self.sub_x).astype(int) + 1
        self.T = (self.time_interval // self.sub_t).astype(int) + 1  # 增加t=0，便于计算初始条件是稳定流动
        self.new = new

        # if new:
        #     self.T += 1

        # self.x_data = 0
        # self.x_data = dataloader.read_field('input')[:, ::sub]
        # self.y_data = dataloader.read_field('output')[:, ::sub_t, ::sub]
        # self.v = dataloader.read_field('visc').item()
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # # gr_sampler = GaussianRF(1, self.time_interval, 2 * math.pi, alpha=2.5, tau=7, device='cpu')
        # # BC_G = gr_sampler.sample(self.n_sample)  # boundary condition
        # G_max = BC_G.max()
        # G_min = BC_G.min()
        # BC_G_data = ((BC_G - G_min) / (G_max - G_min)) * (dataconfig['G_max'] - dataconfig['G_min']) + dataconfig[
        #     'G_min']
        BC_G_data = torch.tensor([[1.0] * 251 + [1.5] * 250] * 2)
        self.BC_G_data = BC_G_data[..., ::self.sub_t]

        # BC_P = gr_sampler.sample(self.n_sample)
        # P_max = BC_P.max()
        # P_min = BC_P.min()
        # BC_P_data = ((BC_P - P_min) / (P_max - P_min)) * (dataconfig['P_max'] - dataconfig['P_min']) + dataconfig[
        #     'P_min']
        BC_P = torch.tensor([300000] * 501)
        self.BC_P_data = BC_P[..., ::self.sub_t]
        # self.x_data = torch.cat((BC_G_data,BC_P_data), dim=1)

    def make_loader(self, config, train=True):
        batch_size = config['train']['batchsize']
        # start = config['data']['offset']
        # Xs = self.x_data[start:start + self.n_sample]
        # ys = self.y_data[start:start + n_sample]

        if self.new:
            # gridx = torch.tensor(np.linspace(0, self.lenth, self.s + 1)[:-1], dtype=torch.float)
            gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float)
            gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)  # 时间没有错吗，难道不是初始条件？
        # else:
        #     gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float)
        #     gridt = torch.tensor(np.linspace(0, 1, self.T + 1)[1:], dtype=torch.float)
        gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float)
        gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)

        # Xs = Xs.reshape(n_sample, 1, self.s).repeat([1, self.T, 1])z
        # self.BC_G_data = torch.cat((self.BC_G_data[:, 0].reshape(self.n_sample, 1), self.BC_G_data), dim=1)
        # self.BC_P_data = torch.cat((self.BC_P_data[:, 0].reshape(self.n_sample, 1), self.BC_P_data), dim=1)
        XsG = self.BC_G_data.reshape(self.n_sample, self.T, 1).repeat([1, 1, self.s])
        XsP = self.BC_P_data.reshape(self.n_sample, self.T, 1).repeat([1, 1, self.s])
        Xs = torch.stack([XsG, XsP, gridx, gridt], dim=3)
        maxmin_normal = maxminnormal(config)
        Xs = maxmin_normal.encode(Xs)
        # dataset = torch.utils.data.TensorDataset(Xs, ys)
        dataset = torch.utils.data.TensorDataset(Xs)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
