from __future__ import print_function
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'experiments'))


import datetime
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import wandb
import utils
from scheduler import WarmupMultiStepLR
from datasets.msr import MSRAction3D
import models.msr as Models
import learning.losses as losses
import argparse
from config.generate_config import load_config
import numpy as np
import torch


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
        for clip, target, video_idx, _ in tqdm(data_loader, desc='Validation' if data_loader.dataset.train else 'Test'):
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

    return total_loss, total_clip_acc, total_video_acc, list_video_class_acc, average_video_class_acc


def main(args):

    config = load_config(args.config)

    wandb.init(project=config['wandb_project'],name=args.config)
    wandb.config.update(config)
    wandb.watch_called = False

    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])    
    device = torch.device(0)

    utils.set_random_seed(config['seed'])

    # Data loading code
    print("Loading data from", config['dataset_path'])
    dataset = MSRAction3D(
        root=config['dataset_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        num_points=config['num_points'],
        train=True,
    )
    dataset_test = MSRAction3D(
        root=config['dataset_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        num_points=config['num_points'],
        train=False,
    )
    
    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'], pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True
    )
    

    print("Number of unique labels (classes):", dataset.num_classes)

    Model = getattr(Models, config['model'])
    print("Creating model:", config['model'])
    if config['model'] == 'P4Transformer':
        model = Model(
            radius=config['radius'],
            nsamples=config['nsamples'],
            spatial_stride=config['spatial_stride'],
            temporal_kernel_size=config['temporal_kernel_size'],
            temporal_stride=config['temporal_stride'],
            emb_relu=config['emb_relu'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            dim_head=config['dim_head'],
            mlp_dim=config['mlp_dim'],
            num_classes=dataset.num_classes
        )
    elif config['model'] == 'PSTNet':
        model = Model(
            radius=config['radius'],
            nsamples=config['nsamples'],
            num_classes=dataset.num_classes)
            
    elif config['model'] == 'PSTNet2':
        model = Model(
            radius=config['radius'],
            nsamples=config['nsamples'],
            num_classes=dataset.num_classes)
    elif config['model'] == 'PSTTransformer':
        model = Model(
            radius=config['radius'],
            nsamples=config['nsamples'],
            spatial_stride=config['spatial_stride'],
            temporal_kernel_size=config['temporal_kernel_size'],
            temporal_stride=config['temporal_stride'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            dim_head=config['dim_head'],
            dropout1=config['dropout1'],
            mlp_dim=config['mlp_dim'],
            num_classes=dataset.num_classes,
            dropout2=config['dropout2']
        )


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    class_weights = utils.compute_class_weights(data_loader, dataset.num_classes)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    if config['loss_type'] == 'std_cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_type'] == 'weighted_cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif config['loss_type'] == 'focal':
        criterion = losses.FocalLoss(alpha=1, gamma=2)
    else:
        raise ValueError("Invalid loss type. Supported types: 'std_cross_entropy', 'weighted_cross_entropy', 'focal'.")

    lr = config['lr']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config['momentum'], weight_decay=config['weight_decay'])
    warmup_iters = config['lr_warmup_epochs'] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config['lr_milestones']]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=config['lr_gamma'], warmup_iters=warmup_iters, warmup_factor=1e-5
    )

    model_without_ddp = model

    if config['resume']:
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model_state_dict = checkpoint['model']

         # Remove the keys related to the last layer from the checkpoint state dictionary before loeading
        last_layer_keys = [key for key in model_state_dict.keys() if key.startswith('mlp_head.3')]  # Assuming the last layer is at index 3
        for key in last_layer_keys:
            del model_state_dict[key]
        model_without_ddp.load_state_dict(model_state_dict, strict=False)  # strict=False allows for partial loadin


        #opt_state_dict = checkpoint['optimizer']

        # Load the optimizer and lr_scheduler state
        #optimizer.load_state_dict(opt_state_dict)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        #args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    acc = 0

    
    for epoch in range(config['start_epoch'], config['epochs']):
        train_clip_loss, train_clip_acc = train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch)
        val_clip_loss, val_clip_acc, val_video_acc, val_list_video_class_acc, val_average_video_class_acc  = evaluate(model, criterion, data_loader_test, device=device)

        wandb.log({
            "Train Loss": train_clip_loss,
            "Train Clip Acc@1": train_clip_acc,
            "Val Loss": val_clip_loss,
            "Val Clip Acc@1": val_clip_acc,
            "Val Video Acc@1": val_video_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "Val Avg Video Class Acc": val_average_video_class_acc  # Log the average per-class accuracy for validation videos
            })

        print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}, Train Acc@1: {train_clip_acc:.4f}")
        print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}, Validation Acc@1: {val_clip_acc:.4f}, Validation Video Acc@1: {val_video_acc:.4f}")
        print(f"Epoch {epoch} - Average Validation Video Class Acc: {val_average_video_class_acc:.4f}")
        print(f"Epoch {epoch} - Validation Video Class Acc: {val_list_video_class_acc.tolist()}")


        if config['output_dir'] and utils.is_main_process():
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': config
            }
            torch.save(
                checkpoint, os.path.join(config['output_dir'], 'model_{}.pth'.format(epoch))
            )
            torch.save(
                checkpoint, os.path.join(config['output_dir'], 'checkpoint.pth')
            ) 
        
        if val_video_acc > acc:
            acc = val_video_acc
            if config['output_dir'] and utils.is_main_process():
                torch.save(
                    checkpoint, os.path.join(config['output_dir'], 'best_model.pth')
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    parser.add_argument('--config', type=str, default='PSTNet2_MSRA/2', help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)
