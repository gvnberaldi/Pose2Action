from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from scripts.scheduler import WarmupMultiStepLR
from learning.losses import FocalLoss
from datasets.bad import BAD
import numpy as np
import scripts.utils as utils
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import os


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    
    header = f'Epoch: [{epoch}]'
    total_clip_acc = 0.0
    total_loss = 0.0

    for batch_idx, (clip, target, _, _) in enumerate(tqdm(data_loader, desc=header)):
        clip, target = clip.to(device), target.to(device)
        output = model(clip)
        loss = criterion(output, target)

        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, _ = utils.accuracy(output, target, topk=(1, 5)) 

        total_clip_acc += acc1.item() * clip.size(0)
        total_loss += loss.item() * clip.size(0)

        # Step the learning rate scheduler
        lr_scheduler.step()

    total_clip_acc /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)

    return total_loss, total_clip_acc


def evaluate(model, criterion, data_loader, device):
    model.eval()
    video_prob = {}
    video_label = {}
    total_clip_acc = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for clip, target, video_idx, _ in tqdm(data_loader, desc='Validation' if data_loader.dataset.mode=="val" else 'Test'):
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            clip_acc, _ = utils.accuracy(output, target, topk=(1, 5))

            total_clip_acc += clip_acc.item() * clip.size(0)
            total_loss += loss.item() * clip.size(0)

            prob = F.softmax(output, dim=1).cpu().numpy()
            target, video_idx = target.cpu().numpy(), video_idx.cpu().numpy()

            for i, idx in enumerate(video_idx):
                video_prob[idx] = video_prob.get(idx, 0) + prob[i]
                video_label[idx] = video_label.get(idx, target[i])

        total_clip_acc /= len(data_loader.dataset)
        total_loss /= len(data_loader.dataset)

        video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
        pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
        total_video_acc = np.mean(pred_correct)*100

        class_count = np.zeros(data_loader.dataset.num_classes)
        class_correct = np.zeros(data_loader.dataset.num_classes)

        for k, v in video_pred.items():
            label = video_label[k]
            class_count[label] += 1
            class_correct[label] += (v == label)

        non_zero_classes = class_count != 0
        list_video_class_acc = class_correct[non_zero_classes] *100 / class_count[non_zero_classes]
        average_video_class_acc = np.mean(list_video_class_acc)

        # Calculate F1 score
        video_true = [video_label[k] for k in video_pred]
        video_pred = list(video_pred.values())
        f1 = f1_score(video_true, video_pred, average='macro')*100

        #add confusion matrix
        conf_matrix= confusion_matrix(video_true, video_pred, labels=None, sample_weight=None, normalize=None)

    return total_loss, total_clip_acc, total_video_acc, list_video_class_acc, average_video_class_acc, f1, conf_matrix

def load_data(config):
    dataset = BAD(
        root=config['data_train_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        max_frame_interval=config['max_frame_interval'],
        num_points=config['num_points'],
        mode="train",
        split_file=config['split_train_path'],
        aug=config['DS_AUGMENTS_CFG']
    )


    dataset_test = BAD(
        root=config['data_test_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        max_frame_interval=config['max_frame_interval'],
        num_points=config['num_points'],
        mode="test",
        split_file=config['split_test_path'],
        aug=config['DS_AUGMENTS_CFG']
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'], pin_memory=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True
    )

 

    if config['val_set']:
        dataset_val = BAD(
            root=config['data_train_path'],
            frames_per_clip=config['clip_len'],
            frame_interval=config['frame_interval'],
            max_frame_interval=config['max_frame_interval'],
            num_points=config['num_points'],
            mode="val",
            split_file=config['split_train_path'],
            aug=config['DS_AUGMENTS_CFG']
    )
        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True
        )
    else:
        data_loader_val = None


    return data_loader, data_loader_test, data_loader_val, dataset.num_classes

def create_criterion(config, data_loader, num_classes, device):
    class_weights = utils.compute_class_weights(data_loader, num_classes)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    loss_type = config.get('loss_type', 'std_cross_entropy')
    if loss_type == 'std_cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
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


def final_test(model, criterion, data_loader, device, output_dir):
    model.eval()
    video_prob = {}
    video_label = {}
    total_clip_acc = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for clip, target, video_idx, _ in tqdm(data_loader, desc='Test'):
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            clip_acc, _ = utils.accuracy(output, target, topk=(1, 5))

            total_clip_acc += clip_acc.item() * clip.size(0)
            total_loss += loss.item() * clip.size(0)

            prob = F.softmax(output, dim=1).cpu().numpy()
            target, video_idx = target.cpu().numpy(), video_idx.cpu().numpy()

            for i, idx in enumerate(video_idx):
                video_prob[idx] = video_prob.get(idx, 0) + prob[i]
                video_label[idx] = video_label.get(idx, target[i])

        total_clip_acc /= len(data_loader.dataset)
        total_loss /= len(data_loader.dataset)

        video_pred = {k: np.argmax(v) for k, v in video_prob.items()}




        pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
        total_video_acc = np.mean(pred_correct)*100

        class_count = np.zeros(data_loader.dataset.num_classes)
        class_correct = np.zeros(data_loader.dataset.num_classes)

        for k, v in video_pred.items():
            label = video_label[k]
            class_count[label] += 1
            class_correct[label] += (v == label)

        non_zero_classes = class_count != 0
        list_video_class_acc = class_correct[non_zero_classes] *100 / class_count[non_zero_classes]

        # Calculate F1 score
        video_true = [video_label[k] for k in video_pred]
        video_pred = list(video_pred.values())
        f1 = f1_score(video_true, video_pred, average='macro', zero_division=1)*100
        report_str = classification_report(video_true, video_pred, zero_division=1)

    return f1, report_str