from __future__ import print_function
import os
import torch
import numpy as np
import torch.utils.data
import torchvision
import argparse
from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils
from scripts import metrics
from tqdm import tqdm

from visualization.plot_pc_joints import gif_gt_out_pc
from trainer_itop import load_data, create_criterion

def evaluate(model, criterion, data_loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    clip_losses = []  # List to store the loss for each clip

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(data_loader, desc='Validation' if data_loader.dataset.train else 'Test'):
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
                clip_losses.append((video_id.cpu().detach().numpy(), loss.item(), clip.cpu().detach().numpy(), target.cpu().detach().numpy(), output.cpu().detach().numpy()))

        total_loss /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches
        total_map /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches
        total_pck /= len(data_loader.dataset)  # Adjusted to divide by total number of clips, not batches

    # Return clip_losses along with the other metrics
    return clip_losses, total_loss, total_map, total_pck

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
    model.to(device)

    criterion = create_criterion(config, data_loader_test, num_classes, device)
    #optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader_test)

    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')

    model_state_dict = checkpoint['model']
    model_without_ddp.load_state_dict(model_state_dict, strict=True)
    #config['start_epoch'] = checkpoint['epoch'] + 1
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #opt_state_dict = checkpoint['optimizer']
    #optimizer.load_state_dict(opt_state_dict)

    losses, val_clip_loss, val_map, val_pck = evaluate(model, criterion, data_loader_test, device=device, threshold=config['threshold'])

    # Sort the clips by loss
    losses.sort(key=lambda x: x[1], reverse=True)

    print(len(losses))
    for i, (video_id, loss, clip, target, output) in enumerate(losses[:100]):
        print("loss: ", loss)
        # Remove the second dimension from clip, target, and output
        clip = np.squeeze(clip, axis=1)
        target = np.squeeze(target, axis=1)
        output = np.squeeze(output, axis=1)

        print(clip.shape)
        print(target.shape)
        print(output.shape)

        gif_gt_out_pc(clip, target, output, i, 'visualization/gifs/test')

    #print(f"Validation Loss: {val_clip_loss:.4f}")
    #print(f"Validation mAP: {val_map:.4f}")
    #print(f"EValidation PCK: {val_pck}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training on ITOP dataset')
    parser.add_argument('--config', type=str, default='P4T_ITOP/36', help='Path to the YAML config file')
    parser.add_argument('--model', type=str, default='experiments/P4T_ITOP/36/log/best_model.pth', help='Path to the YAML config file')

    args = parser.parse_args()
    main(args)