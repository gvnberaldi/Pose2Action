import os

import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch import nn

from SPiKE.datasets.bad import BAD
from scripts.scheduler import WarmupMultiStepLR
from datasets.itop import ITOP
import numpy as np
from scripts import metrics
import const.path as path


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, threshold):
    model.train()

    header = f'Epoch: [{epoch}]'
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    for batch_idx, (clip, target, frame_indices) in enumerate(tqdm(data_loader, desc=header)):
        clip, target = clip.to(device), target.to(device)    
        output = model(clip).reshape(target.shape)
        loss = criterion(output, target)

        pck, map = metrics.joint_accuracy(output, target, threshold)
        total_pck += pck.detach().cpu().numpy()
        total_map += map.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        lr_scheduler.step()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def evaluate(model, criterion, data_loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    with torch.no_grad():
        for clip, target, video_id in tqdm(data_loader, desc='Validation' if data_loader.dataset.train else 'Test'):
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(clip).reshape(target.shape)
            loss = criterion(output, target)

            pck, map = metrics.joint_accuracy(output, target, threshold)
            total_pck += pck.detach().cpu().numpy()
            total_map += map.detach().cpu().item()

            total_loss += loss.item()

        total_loss /= len(data_loader)
        total_map /= len(data_loader)
        total_pck /= len(data_loader)

    return total_loss, total_pck, total_map

def load_data(config):
    if config['dataset'] == 'ITOP':
        dataset = ITOP(
            root=config['dataset_path'],
            frames_per_clip=config['clip_len'],
            frame_interval=config['frame_interval'],
            num_points=config['num_points'],
            train=True,
            use_valid_only=config['use_valid_only'],
            aug_list=config['AUGMENT_TRAIN'],
            label_frame=config['label_frame']
        )

        dataset_test = ITOP(
            root=config['dataset_path'],
            frames_per_clip=config['clip_len'],
            frame_interval=config['frame_interval'],
            num_points=config['num_points'],
            train=False,
            use_valid_only=config['use_valid_only'],
            aug_list=config['AUGMENT_TEST'],
            label_frame=config['label_frame']
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'])
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], num_workers=config['workers'])

    elif config['dataset'] == 'BAD':
        dataset = BAD(root=config['dataset_path'],
                      frames_per_clip=config['clip_len'],
                      frame_interval=config['frame_interval'],
                      num_points=config['num_points'],
                      train=True,
                      labeled_frame=config['label_frame']
                      )

        dataset_test = BAD(root=config['dataset_path'],
                           frames_per_clip=config['clip_len'],
                           frame_interval=config['frame_interval'],
                           num_points=config['num_points'],
                           train=False,
                           labeled_frame=config['label_frame']
                           )

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                                  num_workers=config['workers'])
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'],
                                                       num_workers=config['workers'])

    return data_loader, data_loader_test, dataset.num_coord_joints


def create_criterion(config):
    loss_type = config.get('loss_type', 'std_cross_entropy')

    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError("Invalid loss type. Supported types: 'std_cross_entropy', 'weighted_cross_entropy', 'focal'.")


def create_optimizer_and_scheduler(config, model, data_loader):
    lr = config['lr']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config['momentum'], weight_decay=config['weight_decay'])
    warmup_iters = config['lr_warmup_epochs'] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config['lr_milestones']]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=config['lr_gamma'], warmup_iters=warmup_iters, warmup_factor=1e-5
    )
    return optimizer, lr_scheduler
