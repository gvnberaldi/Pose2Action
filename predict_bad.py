from __future__ import print_function
import os
import torch
import numpy as np
import torch.utils.data
import torchvision
import argparse

from torch.utils.data import DataLoader

from SPiKE.datasets.bad import BAD
from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils
from scripts import metrics
from tqdm import tqdm
import random

from visualization.plot_pc_joints import gif_gt_out_pc, clean_gif_gt_out_pc, clean_sep_gt_out_pc
from trainer import load_data, create_criterion


def evaluate(model, criterion, data_loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    clip_losses = []  # List to store the loss for each clip

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(data_loader,
                                                                desc='Validation' if data_loader.dataset.train else 'Test'):
            for i in range(len(batch_clips)):
                clip = batch_clips[i].unsqueeze(0).to(device, non_blocking=True)
                target = batch_targets[i].unsqueeze(0).to(device, non_blocking=True)
                video_id = batch_video_ids[i]

                output = model(clip).reshape(target.shape)
                loss = criterion(output, target)

                pck, map = metrics.joint_accuracy(output, target, threshold)
                total_pck += pck.detach().cpu().numpy()
                total_map += map.detach().cpu().item()

                total_loss += loss.item()

                # Convert tensors to numpy arrays and append the loss, ground truth, and labels for this clip to clip_losses
                clip_losses.append((video_id.cpu().detach().numpy(), loss.item(), clip.cpu().detach().numpy(),
                                    target.cpu().detach().numpy(), output.cpu().detach().numpy()))

        total_loss /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches
        total_map /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches
        total_pck /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches

    # Return clip_losses along with the other metrics
    return clip_losses, total_loss, total_map, total_pck


def main(args):
    config = load_config(args.config)

    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)
    print("CUDA version:", torch.version.cuda)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])
    # print("CUDA_VISIBLE_DEVICES: ", os.environ["module list"])
    device = torch.device(0)
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the name of the GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU name: {gpu_name}")
    else:
        print("CUDA is not available.")

    utils.set_random_seed(config['seed'])

    # Data loading code
    print("Loading data from", config['dataset_path'])
    data_loader, data_loader_test, num_classes = load_data(config)
    print("Number of unique labels (classes):", num_classes)

    print("Number of unique labels (classes):", num_classes)

    model = model_factory.create_model(config, num_classes)
    model_without_ddp = model
    model.to(device)

    criterion = create_criterion(config)
    # optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader_test)

    print(f"Loading model from {config['resume']}")
    checkpoint = torch.load(config['resume'], map_location='cpu')

    model_state_dict = checkpoint['model']
    model_without_ddp.load_state_dict(model_state_dict, strict=True)
    # config['start_epoch'] = checkpoint['epoch'] + 1
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # opt_state_dict = checkpoint['optimizer']
    # optimizer.load_state_dict(opt_state_dict)

    losses, val_clip_loss, val_map, val_pck = evaluate(model, criterion, data_loader_test, device=device,
                                                       threshold=config['threshold'])

    # Shuffle the clips
    losses.sort(key=lambda x: x[1], reverse=True)
    # random.shuffle(losses)

    print(len(losses))
    for i, (video_id, loss, clip, target, output) in enumerate(losses):
        print("loss: ", loss)
        # Remove the second dimension from clip, target, and output
        clip = np.squeeze(clip, axis=0)
        target = np.squeeze(target, axis=0)
        output = np.squeeze(output, axis=0)

        print(f"Clip shape: {clip.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Model output shape: {output.shape}")

        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "visualization/gifs/BAD")
        os.makedirs(output_dir, exist_ok=True)

        gif_gt_out_pc(clip, target, output, video_id, output_directory=output_dir, label_frame='middle')

    print(f"Validation Loss: {val_clip_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"EValidation PCK: {val_pck}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPiKE Model Training on BAD dataset')
    parser.add_argument('--config', type=str, default='BAD', help='Path to the YAML config file')

    args = parser.parse_args()
    main(args)

