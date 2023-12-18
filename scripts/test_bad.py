from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
import utils

import torch.nn.functional as F

from sklearn.metrics import f1_score

from scheduler import WarmupMultiStepLR

from data.datasets.bad import BAD
import models.msr as Models

from sklearn.metrics import confusion_matrix

import learning.losses as losses
import argparse
from config.config_test import load_config, construct_data_paths


def evaluate(model, criterion, data_loader, device):
    model.eval()
    video_prob = {}
    video_label = {}

    clip_predictions = []
    clip_targets = []

    video_predictions = []
    video_labels = []

    class_f1_scores = []  # Add this line to store F1-score for each class


    with torch.no_grad():
        total_clip_acc1 = 0.0
        total_clip_acc5 = 0.0
        total_loss = 0.0
        clip_f1 = 0.0 

        for clip, target, video_idx, _ in tqdm(data_loader, desc='Validation' if data_loader.dataset.train else 'Test'):

            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            clip_acc1, clip_acc5 = utils.accuracy(output, target, topk=(1, 5))

            total_clip_acc1 += clip_acc1.item() * clip.size(0)
            total_clip_acc5 += clip_acc5.item() * clip.size(0)
            total_loss += loss.item() * clip.size(0)

            prob = F.softmax(input=output, dim=1)

            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            
            # Collect clip-level predictions and targets
            clip_predictions.extend(np.argmax(prob, axis=1))
            clip_targets.extend(target)

        total_clip_acc1 /= len(data_loader.dataset)
        total_clip_acc5 /= len(data_loader.dataset)
        total_loss /= len(data_loader.dataset)

        # Calculate F1 score at the clip level
        clip_f1 = f1_score(clip_targets, clip_predictions, average='macro')

        #Get the confusion matrix
        conf_matrix_clip= confusion_matrix(clip_targets, clip_predictions)

        # Compute accuracy per class
        clip_class_accuracy = conf_matrix_clip.diagonal() / conf_matrix_clip.sum(axis=1)

        # video level prediction
        video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
        pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
        total_video_acc = np.mean(pred_correct)

        class_count = [0] * data_loader.dataset.num_classes
        class_correct = [0] * data_loader.dataset.num_classes

        for k, v in video_pred.items():
            label = video_label[k]
            class_count[label] += 1
            class_correct[label] += (v==label)
        
        # Filter out classes with a count of zero
        non_zero_classes = [(c, s) for c, s in zip(class_correct, class_count) if s != 0]

        # Calculate class_acc only for non-zero classes
        class_acc = [c / float(s) for c, s in non_zero_classes]

        # Calculate F1 score at the video level
        video_predictions = [video_pred[k] for k in video_pred]
        video_labels = [video_label[k] for k in video_label]
        video_f1 = f1_score(video_labels, video_predictions, average='macro') 

        #Get the confusion matrix
        conf_matrix_video = confusion_matrix(video_labels, video_predictions)

        # Compute accuracy per class
        video_class_accuracy = conf_matrix_video.diagonal() / conf_matrix_video.sum(axis=1)

        return total_loss, total_clip_acc1, total_clip_acc5, total_video_acc, clip_f1, video_f1, conf_matrix_clip, conf_matrix_video, clip_class_accuracy, video_class_accuracy

def main(args):

    config = load_config(args.config)
    config = construct_data_paths(config)

    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])

    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(0)

    # Data loading code
    print("Loading data")
    dataset = BAD(
        root=config['data_train_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        max_frame_interval=config['max_frame_interval'],
        num_points=config['num_points'],
        train=True,
        split_file=config['split_train_path'],
        aug=[]
    )

    dataset_test = BAD(
        root=config['data_test_path'],
        frames_per_clip=config['clip_len'],
        frame_interval=config['frame_interval'],
        max_frame_interval=config['max_frame_interval'],
        num_points=config['num_points'],
        train=False,
        split_file=config['split_test_path'],
        aug=[]
    )

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'], pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True
    )
    
    # Extract the number of unique labels (classes) from the dataset
    num_classes = dataset.num_classes
    print("Number of unique labels (classes):", num_classes)

    print("Creating model")
    Model = getattr(Models, config['model'])
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
    model.to(device)

    parameters = utils.count_parameters(model)

    num_classes = dataset.num_classes 
    class_weights = utils.compute_class_weights(data_loader, num_classes)
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
        model_without_ddp.load_state_dict(model_state_dict, strict=False)  # strict=False allows for partial loadin


    print("Start testing")
    #unfreeze_interval = 10  # Unfreeze every 10 epochs


    # Unfreeze additional layers every unfreeze_interval epochs
    #    if epoch % unfreeze_interval == 0 and epoch > 0:
    #        num_layers_to_unfreeze = min((epoch // unfreeze_interval) + 1, len(model.transformer.layers))
    #        model.freeze_transformer_layers(num_layers_to_unfreeze)

    val_clip_loss, val_clip_acc1, val_clip_acc5, val_video_acc, clip_f1, video_f1, conf_matrix_clip, conf_matrix_video, clip_class_accuracy, video_class_accuracy  = evaluate(model, criterion, data_loader_test, device=device)


    print("val_clip_loss:", val_clip_loss)
    print("val_clip_acc1:", val_clip_acc1)
    print("val_clip_acc5:", val_clip_acc5)
    print("val_video_acc:", val_video_acc)
    print("clip_f1:", clip_f1)
    print("video_f1:", video_f1)
    print("conf_matrix_clip:", conf_matrix_clip)
    print("conf_matrix_video:", conf_matrix_video)
    print("clip_class_accuracy:", clip_class_accuracy)
    print("video_class_accuracy:", video_class_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    parser.add_argument('--config', type=str, default='config/config_test.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)
