from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scripts.scheduler import WarmupMultiStepLR
from datasets.synthia import *

import numpy as np
import scripts.utils as utils


def train_one_epoch(
    model, criterion, optimizer, lr_scheduler, data_loader, device, epoch
):
    model.train()

    header = f"Epoch: [{epoch}]"

    loss = 0.0

    for batch_idx, (pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2) in enumerate(
        tqdm(data_loader, desc=header)
    ):
        pc1, rgb1, label1, mask1 = (
            pc1.to(device),
            rgb1.to(device),
            label1.to(device),
            mask1.to(device),
        )
        output1 = model(pc1, rgb1).transpose(1, 2)
        loss1 = criterion(output1, label1) * mask1
        loss1 = torch.sum(loss1) / (torch.sum(mask1) + 1)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        pc2, rgb2, label2, mask2 = (
            pc2.to(device),
            rgb2.to(device),
            label2.to(device),
            mask2.to(device),
        )
        output2 = model(pc2, rgb2).transpose(1, 2)
        loss2 = criterion(output2, label2) * mask1
        loss2 = torch.sum(loss2) / (torch.sum(mask2) + 1)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        loss = (loss1.item() + loss2.item()) / 2.0

        lr_scheduler.step()

    return loss


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_seen = 0
    total_pred_class = [0] * 12
    total_correct_class = [0] * 12
    total_class = [0] * 12

    with torch.no_grad():
        for pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2 in tqdm(data_loader, desc='Validation' if data_loader.dataset.train else 'Test'):
            pc1, rgb1 = pc1.to(device), rgb1.to(device)
            output1 = model(pc1, rgb1).transpose(1, 2)
            loss1 = criterion(output1, label1.to(device)) * mask1.to(device)
            loss1 = torch.sum(loss1) / (torch.sum(mask1.to(device)) + 1)
            label1, mask1 = label1.numpy().astype(np.int32), mask1.numpy().astype(
                np.int32
            )
            output1 = output1.cpu().numpy()
            pred1 = np.argmax(output1, 1)  # BxTxN
            correct1 = np.sum((pred1 == label1) * mask1)
            total_correct += correct1
            total_seen += np.sum(mask1)
            for c in range(12):
                total_pred_class[c] += np.sum(((pred1 == c) | (label1 == c)) & mask1)
                total_correct_class[c] += np.sum((pred1 == c) & (label1 == c) & mask1)
                total_class[c] += np.sum((label1 == c) & mask1)

            pc2, rgb2 = pc2.to(device), rgb2.to(device)
            output2 = model(pc2, rgb2).transpose(1, 2)
            loss2 = criterion(output2, label2.to(device)) * mask2.to(device)
            loss2 = torch.sum(loss2) / (torch.sum(mask2.to(device)) + 1)
            label2, mask2 = label2.numpy().astype(np.int32), mask2.numpy().astype(
                np.int32
            )
            output2 = output2.cpu().numpy()
            pred2 = np.argmax(output2, 1)  # BxTxN
            correct2 = np.sum((pred2 == label2) * mask2)
            total_correct += correct2
            total_seen += np.sum(mask2)
            for c in range(12):
                total_pred_class[c] += np.sum(((pred2 == c) | (label2 == c)) & mask2)
                total_correct_class[c] += np.sum((pred2 == c) & (label2 == c) & mask2)
                total_class[c] += np.sum((label2 == c) & mask2)

            total_loss += (loss1.item() + loss2.item()) / 2.0

    # Calculate and return metrics
    ACCs = [
        total_correct_class[c] / float(total_class[c]) if total_class[c] != 0 else 0
        for c in range(12) #TODO: change this number of classes
    ]
    
    #This is not calculating the mIoU correctly!!! MeteorNet does do it either
    mIoUs = [
        total_correct_class[c] / float(total_pred_class[c])
        if total_pred_class[c] != 0
        else 0
        for c in range(12)  #TODO: change this number of classes
    ]

    eval_accuracy = np.mean(np.array(ACCs))
    eval_mIoU = np.mean(np.array(mIoUs))
    eval_loss = total_loss / len(data_loader)

    return eval_accuracy, eval_mIoU, eval_loss


def create_optimizer_and_scheduler(config, model, data_loader):
    lr = config["lr"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    warmup_iters = config["lr_warmup_epochs"] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config["lr_milestones"]]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=config["lr_gamma"],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )
    return optimizer, lr_scheduler


def load_data(config):
    dataset = SegDataset(
        root=config["dataset_path"],
        meta=config["data_eval"],
        labelweight=config["label_weights"],
        frames_per_clip=config["clip_len"],
        num_points=config["num_points"],
        train=True,
    )
    dataset_test = SegDataset(
        root=config["dataset_path"],
        meta=config["data_train"],
        labelweight=config["label_weights"],
        frames_per_clip=config["clip_len"],
        num_points=config["num_points"],
        train=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        pin_memory=True,
    )
    return data_loader, data_loader_test
