import os
import datetime
import time
import torch
import sys
import torch.utils.data
from torch import nn
import wandb
import argparse

sys.path.append(os.getcwd())
os.chdir(os.getcwd())

from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils

from trainer import train_one_epoch, evaluate, load_data, create_criterion, create_optimizer_and_scheduler


def main(args):

    config = load_config(args.config)
    print("torch version: ", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])
    # print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the name of the GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU name: {gpu_name}")
    else:
        print("CUDA is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading code
    print("Loading data from", config['dataset_path'])
    data_loader, data_loader_test, num_coord_joints = load_data(config)

    model = model_factory.create_model(config, num_coord_joints)
    model_without_ddp = model

    # print(f"Number of parameters: {count_parameters(model)}")

    if torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = create_criterion(config)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)


    if config['resume']:
        # Load the pre-trained state_dict
        checkpoint = torch.load(config['resume'], map_location='cpu')
        pretrained_dict = checkpoint['model']
        # Create a new state_dict with renamed keys
        new_pretrained_dict = {}
        for key, value in pretrained_dict.items():
            new_key = key
            if key == 'tube_embedding.conv_d.0.weight':
                new_key = 'stem.conv_d.0.weight'
            elif key == 'pos_embedding.weight':
                new_key = 'pos_embed.weight'
            elif key == 'pos_embedding.bias':
                new_key = 'pos_embed.bias'
            new_pretrained_dict[new_key] = value
        checkpoint['model'] = new_pretrained_dict

        print(f"Loading model from {config['resume']}")
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        config['start_epoch'] = checkpoint['epoch'] + 1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    wandb.login(key='e5e7b3c0c3fbc088d165766e575853c01d6cb305')
    wandb.init(project=config['wandb_project'], entity='gvnberaldi')
    wandb.config.update(config)
    wandb.watch_called = False

    print("Start training")
    print(f"Num epochs: {config['epochs']}")
    start_time = time.time()
    min_loss = sys.maxsize
    eval_thresh = config['threshold']

    if config['output_dir']:
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

    print(f"Train batches: {len(data_loader)}")
    print(f"Test batches: {len(data_loader_test)}")

    for epoch in range(0, 100):
        train_clip_loss, train_pck, train_map = train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                                                                device, epoch, eval_thresh)
        val_clip_loss, val_pck, val_map = evaluate(model, criterion, data_loader_test, device=device,
                                                   threshold=eval_thresh)

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

        print(f"Is main process: {utils.is_main_process()}")
        if config['output_dir'] and utils.is_main_process():
            print("Saving checkpoint...")
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

            torch.save(checkpoint, os.path.join(output_dir, 'from_scratch_checkpoint.pth'))

            if val_clip_loss < min_loss:
                print("Saving model weights...")
                min_loss = val_clip_loss
                torch.save(checkpoint, os.path.join(output_dir, 'from_scratch_best_model.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPiKE Training on BAD dataset')
    parser.add_argument('--config', type=str, default='BAD',
                        help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)
