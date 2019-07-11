#coding:utf-8
import path
from common.h36m_dataset import CMUPanoDataset
from common.data_utils import fetch, read_3d_data, create_2d_data, get_MUCO3DHP_data
from common.graph_utils import adj_mx_from_skeleton
from models.sem_gcn import MultiSemGCN
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import datetime
import os
from common.log import Logger, savefig
from torch.utils.data import DataLoader
from common.utils import AverageMeter, lr_decay, save_ckpt
import time
from progress.bar import Bar
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
def main():
    human36m_data_path = ""
    MUCO3DHP_path = ""
    hid_dim = 128
    num_layers = 4
    non_local = True
    lr = 1.0e-3
    epochs = 30
    _lr_decay = 100000
    lr_gamma = 0.96
    max_norm = True
    num_workers = 8
    snapshot = 5
    batch_size = 64
    print('==> Loading multi-person dataset...')
    human36m_dataset_path = path.join(human36m_data_path)
    dataset = CMUPanoDataset(human36m_dataset_path)
    data_2d, data_3d, img_name = get_MUCO3DHP_data(MUCO3DHP_path)  ## N * (M*17) * 2    N * (M*17) * 3 numpy

    ### divide into trainsets and testsets 4/5 and 1/5
    num = len(data_2d)
    train_num = num * 4 / 5

    cudnn.benchmark = True
    device = torch.device("cuda")

    adj, adj_mutual = adj_mx_from_skeleton(dataset.skeleton()) #ok
    model_pos = MultiSemGCN(adj, adj_mutual, hid_dim, num_layers=num_layers,
                       nodes_group=dataset.skeleton().joints_group() if non_local else None).to(device)  #ok
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=lr)

    start_epoch = 0
    error_best = None
    glob_step = 0
    lr_now = lr
    ckpt_dir_path = path.join('checkpoint',
                              datetime.datetime.now().isoformat() + "_l_%04d_hid_%04d_e_%04d_non_local_%d" % (
                              num_layers, hid_dim, epochs, non_local))

    if not path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)
        print('=> Making checkpoint dir: {}'.format(ckpt_dir_path))

    logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
    logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])

    train_loader = DataLoader(PoseGenerator(data_3d[:train_num], data_2d[:train_num]), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    valid_loader = DataLoader(PoseGenerator(data_3d[train_num:], data_2d[train_num:]), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, lr, lr_now,
                                              glob_step, _lr_decay, lr_gamma, max_norm=max_norm)

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step

def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d).cpu()
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg