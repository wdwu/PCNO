from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from .losses import LpLoss, PINO_loss
from .utils import save_checkpoint, calculate_weight

try:
    import wandb
except ImportError:
    wandb = None


def train_2d_network(model,
                     train_loader,
                     optimizer, scheduler,
                     config,
                     rank=0, log=False,
                     project='PINO-2d-default',
                     group='default',
                     tags=['default'],
                     use_tqdm=True,
                     device=None):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    model.train()
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    fNS = config['train']['fNS_loss']
    fen = config['train']['fen_loss']
    BC_m = config['train']['BC_loss']
    IC_m = config['train']['IC_loss']
    # torch.autograd.set_detect_anomaly(True) #对正向传播进行nan检查
    writer = SummaryWriter()
    # writer.add_graph(model=model)
    for e in pbar:
        model.train()
        train_f_energy = 0.0
        train_f_NS = 0.0
        train_BC = 0.0
        train_IC = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for x in train_loader:
            # x, y = x.to(rank), y.to(rank)
            x = x[0].to(device)
            # out = model(x).reshape(y.shape)
            out = model(x)
            # data_loss = myloss(out, y)

            loss_BC, loss_IC, loss_f_en, loss_f_NS = PINO_loss(out, x, config)  # 仅放入batches个初始条件
            IC_w, BC_w, fen_w, fNS_w = calculate_weight(e, config, loss_BC, loss_IC, num_epochs_rampup=0)
            total_loss = loss_IC * IC_w + loss_BC * BC_w + loss_f_en * fen_w + loss_f_NS * fNS_w
            data_loss = loss_IC * IC_m + loss_BC * BC_m + loss_f_en * fen + loss_f_NS * fNS
            # lossf: L_PDE loss  Lossu: L_PDE
            optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            total_loss.backward()
            # 应用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)

            optimizer.step()
            train_f_energy += loss_f_en.item()
            train_f_NS += loss_f_NS.item()
            train_BC += loss_BC.item()
            train_IC += loss_IC.item()
            train_loss += total_loss.item()
            data_l2 += data_loss.item()
        scheduler.step()
        data_l2 /= len(train_loader)
        train_f_energy /= len(train_loader)
        train_f_NS /= len(train_loader)
        train_BC /= len(train_loader)
        train_IC /= len(train_loader)
        train_loss /= len(train_loader)

        # 用tensorboard记录loss，学习率，权重和梯度
        writer.add_scalars('Learning Curve', {'Train Loss': train_loss}, e)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], e)
        for name, param in model.named_parameters():
            writer.add_scalar("weights/" + name, param.norm(), e)
            writer.add_scalar("gradients/" + name, param.grad.norm(), e)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.6f} '
                    f'train f_en error: {train_f_energy:.6f}; '
                    f'train f_NS error: {train_f_NS:.6f}; '
                    f'train f_BC error: {train_BC:.6f}; '
                    f'train f_IC error: {train_IC:.6f}; '
                    f'data l2 error: {data_l2:.6f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_f_energy,
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
    writer.flush()
    writer.close()
    print('Done!')
