import math

import torch
import torch.nn.functional as F

from train_utils.utils import maxminnormal, convolution


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def PINO_loss(uu, u0, config, difference='implicit difference'):
    u = uu.clone()  # 防止在利用输出计算loss时改变了输出结果，从而无法继续backward
    dataconfig = config['data']
    lenth = dataconfig['f_lenth']
    time_interval = dataconfig['time_interval']
    D = dataconfig['f_D']
    Z = dataconfig['f_Z']
    R = dataconfig['f_R']
    T = dataconfig['f_T']
    frac = dataconfig['f_fraction']
    A = math.pi * D ** 2 / 4  # pipeline cross sectional area
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    # 解码
    maxmin_normal = maxminnormal(config)
    u = maxmin_normal.decode(u)
    u0 = maxmin_normal.decode(u0)

    uG = u[:, :, :, 0]
    uP = u[:, :, :, 1]
    Pv = uG / A  # (1,251,61)
    Pou = uP / (Z * R * T)  # (1,251,61)
    V = Pv / Pou  # (1,251,61)

    # 边界条件
    BC_G = u0[:, :, -1, 0]  # 末端流量
    BC_P = u0[:, :, 0, 1]  # 首端压力
    BC1 = F.mse_loss(u[:, :, -1, 0] / A, BC_G / A)  # 末端流量
    BC2 = F.mse_loss(u[:, :, 0, 1] / (Z * R * T), BC_P / (Z * R * T))  # 首端压力
    loss_BC = BC1 + BC2

    # 初始条件
    # 初始稳定流动
    IC1 = F.mse_loss(uG[:, 0, :] / A, BC_G[:, 0].repeat(1, nx) / A)  # G(x,0)=BC
    ICp1 = -frac * BC_G[:, 0] / A * torch.abs(BC_G[:, 0] / A) / (D * Z * R * T)  #
    ICp2 = torch.linspace(0, lenth, nx).reshape(1, nx).to(u.device)
    ICp = ICp1 * ICp2 + (BC_P[:, 0] / (Z * R * T)) ** 2
    IC2 = F.mse_loss(ICp, Pou[:, 0, :] ** 2)
    loss_IC = IC1 + IC2

    # 求导
    dx = lenth / (nx - 1)
    dt = time_interval / (nt - 1)
    if difference == 'both':
        ux = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)  # 中心差分
        ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    elif difference == 'forward':
        ux = (u[:, :, 1:] - u[:, :, :-1]) / (dx)  # 前向差分
        ut = (u[:, 1:, :] - u[:, :-1, :]) / (dt)
    if difference != 'implicit difference':
        ux_Pv = ux[..., 0] / A
        ux_P = ux[..., 1]
        ut_Pv = ut[..., 0] / A
        ut_Pou = ut[..., 1] / (Z * R * T)

    # 方程
    if difference == 'both':
        # energy function
        en = ut_Pou[:, :, 1:-1] + ux_Pv[:, 1:-1, :]
        # NS function
        NS = ut_Pv[:, :, 1:-1] + ux_P[:, 1:-1, :] + frac * (Pv * torch.abs(V))[:, 1:-1, 1:-1] / (2 * D)

    elif difference == 'forward':
        # energy function
        en = ut_Pou[:, :, :-1] + ux_Pv[:, :-1, :]
        # NS function
        NS = ut_Pv[:, :, :-1] + ux_P[:, :-1, :] + frac * (Pv * torch.abs(V))[:, :-1, :-1] / (2 * D)

    elif difference == 'implicit difference':
        en = convolution(Pou, 't') + dt / dx * convolution(Pv, 'x')
        NS = convolution(Pv, 't') + dt / dx * convolution(uP, 'x') + \
             frac * dt / (4 * D) * convolution(Pv * abs(V), 'ones')

    f1 = torch.zeros(en.shape, device=u.device)
    f2 = torch.zeros(NS.shape, device=u.device)
    loss_en = F.mse_loss(en, f1)
    loss_NS = F.mse_loss(NS, f2)
    return loss_BC, loss_IC, loss_en, loss_NS

    # BC_G = u0[:, :, -1, 0]  # 末端流量
    # BC_P = u0[:, :, 0, 1]  # 首端压力
    #
    # # 初始条件
    # # 初始稳定流动
    ## IC2 = (F.mse_loss(uG[:, 0, :], torch.tensor([1.5] * nx).reshape(1, nx).to(u.device)))  # G(x,0)=BC
    # IC2 = (F.mse_loss(u[:, 0, :, 0], BC_G[:, 0].repeat(1, nx)))  # G(x,0)=BC
    # ICp1 = -frac * BC_G[:, 0] * torch.abs(BC_G[:, 0]) / (A ** 2 * D)  #
    # ICp2 = torch.linspace(0, lenth, nx).reshape(1, nx).to(u.device)
    # ICp = ICp1 * ICp2 + BC_P[:, 0] ** 2 / (R * T)
    # IC4 = F.mse_loss(torch.sqrt(ICp), torch.sqrt(u[:, 0, :, 1] ** 2 / (R * T)))
    # loss_IC = IC2 + IC4
    # loss_IC = torch.sqrt(loss_IC + 1e-8)
    #
    # # 求导
    # dx = lenth / (nx - 1)
    # dt = time_interval / (nt - 1)
    # if difference == 'both':
    #     ux = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)  # 双向差分
    #     ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    # elif difference == 'forward':
    #     ux = (u[:, :, 1:] - u[:, :, :-1]) / (dx)  # 前向差分
    #     ut = (u[:, 1:, :] - u[:, :-1, :]) / (dt)
    #
    # ux_G = ux[..., 0]
    # ux_P = ux[..., 1]
    # ut_G = ut[..., 0]
    # ut_P = ut[..., 1]
    #
    # # 方程
    # if difference == 'both':
    #     # energy function
    #     loss_en = ut_P[:, :, 1:-1] * A + ux_G[:, 1:-1, :] * (R * T)
    #     # NS function
    #     loss_NS = frac * R * T * (uG * torch.abs(uG))[:, 1:-1, 1:-1] / (2 * A * D)
    #     loss_NS += (ut_G[:, :, 1:-1] + ux_P[:, 1:-1, :] * A) * uP[:, 1:-1, 1:-1]
    #
    # elif difference == 'forward':
    #     # energy function
    #     loss_en = ut_P[:, :, 0:-1] * A + ux_G[:, 0:-1, :] * (R * T)
    #     # NS function
    #     loss_NS = frac * (uG * torch.abs(uG))[:, 0:-1, 0:-1] / (2 * A * D)
    #     loss_NS += (ut_G[:, :, 0:-1] + ux_P[:, 0:-1, :] * A) * uP[:, 0:-1, 0:-1] / (R * T)
    #
    # f1 = torch.zeros(loss_en.shape, device=u.device)
    # f2 = torch.zeros(loss_NS.shape, device=u.device)
    # loss_en = torch.sqrt(F.mse_loss(loss_en, f1) + 1e-8)
    # loss_NS = torch.sqrt(F.mse_loss(loss_NS, f2) + 1e-8)

    # return loss_BC, loss_IC, loss_en, loss_NS
