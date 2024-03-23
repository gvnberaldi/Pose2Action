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

from trainer_itop import train_one_epoch, evaluate, load_data, create_criterion, create_optimizer_and_scheduler, \
    final_test


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
    data_loader, data_loader_test, num_classes = load_data(config)
    print("Number of unique labels (classes):", num_classes)

    model = model_factory.create_model(config, num_classes)
    model_without_ddp = model

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = create_criterion(config, data_loader, num_classes, device)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

    if config['resume']:  # TODO: check if this
        print(f"Loading model from {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model_state_dict = checkpoint['model']
        model_without_ddp.load_state_dict(model_state_dict, strict=True)
        config['start_epoch'] = checkpoint['epoch'] + 1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)

    wandb.init(project=config['wandb_project'], name=args.config)
    wandb.config.update(config)
    wandb.watch_called = False
 
    print("Start training")
    start_time = time.time()
    min_loss = sys.maxsize
    eval_thresh = config['threshold']

    for epoch in range(config['start_epoch'], config['epochs']):
        train_clip_loss, train_pck, train_map = train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, eval_thresh)
        val_clip_loss, val_pck, val_map = evaluate(model, criterion, data_loader_test, device=device, threshold=eval_thresh)

        data1 = [(idx, train_pck[idx]) for idx in range(len(train_pck))]
        data2 = [(idx, val_pck[idx]) for idx in range(len(val_pck))]

        table1 = wandb.Table(data=data1, columns=["joint", "pck"])
        table2 = wandb.Table(data=data2, columns=["joint", "pck"])

        wandb.log({
            "Train loss": train_clip_loss,
            "Train mAP": train_map,
            "Train PCK": wandb.plot.bar(table1, "joint", "pck", title="Train PCK"),
            "Val loss": val_clip_loss,
            "Val mAP": val_map,
            "Val PCK": wandb.plot.bar(table2, "joint", "pck", title="Val PCK"),
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}")
        print(f"Epoch {epoch} - Train mAP: {train_map:.4f}")
        print(f"Epoch {epoch} - Train PCK: {train_pck}")
        print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}")
        print(f"Epoch {epoch} - Validation mAP: {val_map:.4f}")
        print(f"Epoch {epoch} - Validation PCK: {val_pck}")

        if config['output_dir'] and utils.is_main_process():
            # If the model is wrapped in DataParallel or DistributedDataParallel, unwrap it
            model_to_save = model_without_ddp.module if isinstance(model_without_ddp,
                                                                   (nn.DataParallel)) else model_without_ddp

            checkpoint = {
                'model': model_to_save.state_dict(),  # Save the unwrapped model's state dictionary
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': config
            }
            torch.save(
                checkpoint, os.path.join(config['output_dir'], 'checkpoint.pth')
            )

        if val_clip_loss < min_loss:
            min_loss = val_clip_loss
            if config['output_dir'] and utils.is_main_process():
                torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training on ITOP dataset')
    parser.add_argument('--config', type=str, default='P4T_ITOP/37-b-w', help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)