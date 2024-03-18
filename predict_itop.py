from __future__ import print_function
import os
import datetime
import time
import torch
import sys
import numpy as np
import torch.utils.data
from torch import nn
import torchvision
import wandb
import argparse
from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils

from trainer_itop import evaluate, load_data, create_criterion, create_optimizer_and_scheduler

def main(args):
    config = load_config(args.config)

    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device(0)

    utils.set_random_seed(config['seed'])

    # Data loading code
    print("Loading data from", config['dataset_path'])
    _, data_loader_test, num_classes = load_data(config)
    print("Number of unique labels (classes):", num_classes)

    model = model_factory.create_model(config, num_classes)
    model_without_ddp = model
    model.to(device)

    criterion = create_criterion(config, data_loader_test, num_classes, device)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader_test)



    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')

    model_state_dict = checkpoint['model']
    model_without_ddp.load_state_dict(model_state_dict, strict=True)
    config['start_epoch'] = checkpoint['epoch'] + 1
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    opt_state_dict = checkpoint['optimizer']
    optimizer.load_state_dict(opt_state_dict)

    eval_thresh = config['threshold']

    val_clip_loss, val_pck, val_map = evaluate(model, criterion, data_loader_test, device=device, threshold=eval_thresh)


    print(f"Validation Loss: {val_clip_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"EValidation PCK: {val_pck}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training on ITOP dataset')
    parser.add_argument('--config', type=str, default='P4T_ITOP/36', help='Path to the YAML config file')
    parser.add_argument('--model', type=str, default='experiments/P4T_ITOP/36/log/best_model.pth', help='Path to the YAML config file')

    args = parser.parse_args()
    main(args)