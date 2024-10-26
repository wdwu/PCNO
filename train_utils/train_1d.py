import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None


def train_1d_network(model,
                     a_loader,
                     optimizer, scheduler,
                     config,
                     rank=0, log=False,
                     project='PINO-2d-default',
                     group='default',
                     tags=['default'],
                     use_tqdm=True):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    # data_weight = config['train']['xy_loss']
    # f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for x in a_loader:
            # x, y = x.to(rank), y.to(rank)
            x = x.to(rank)
            out = model(x)
            # data_loss = myloss(out, y)

            loss_u, loss_f = PINO_loss(out, x, config)  # 仅放入batches个初始条件
            total_loss = loss_u * ic_weight + loss_f * f_weight + data_loss * data_weight  # lossu：#初始时刻的训练输出u-给定的初始条件u0
            # lossf: L_PDE loss  Lossu: L_PDE
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
        scheduler.step()
        data_l2 /= len(a_loader)
        train_pino /= len(a_loader)
        train_loss /= len(a_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_pino,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')
