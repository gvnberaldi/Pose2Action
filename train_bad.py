from __future__ import print_function
import os
import datetime
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import wandb
import argparse
from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils



from trainer_bad import train_one_epoch, evaluate, load_data, create_criterion, create_optimizer_and_scheduler, final_test

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
    print("Loading data from", config['dataset_root'])
    data_loader, data_loader_test, num_classes = load_data(config)    
    print("Number of unique labels (classes):", num_classes)
    
    model = model_factory.create_model(config, num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using", torch.cuda.device_count(), "GPUs!")

    model.to(device)

    criterion = create_criterion(config, data_loader, num_classes, device)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

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
        val_clip_loss, val_clip_acc, val_video_acc, val_list_video_class_acc, val_average_video_class_acc, f1, confusion_matrix  = evaluate(model, criterion, data_loader_test, device=device)

        wandb.log({
            "Train Loss": train_clip_loss,
            "Train Clip Acc@1": train_clip_acc,
            "Val Loss": val_clip_loss,
            "Val Clip Acc@1": val_clip_acc,
            "Val Video Acc@1": val_video_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "Val Avg Video Class Acc": val_average_video_class_acc,  # Log the average per-class accuracy for validation videos
            "F1": f1
            })

        print(f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}, Train Acc@1: {train_clip_acc:.4f}")
        print(f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}, Validation Acc@1: {val_clip_acc:.4f}, Validation Video Acc@1: {val_video_acc:.4f}")
        print(f"Epoch {epoch} - Average Validation Video Class Acc: {val_average_video_class_acc:.4f}")
        print(f"Epoch {epoch} - Validation Video Class Acc: {val_list_video_class_acc.tolist()}")
        print(f"Epoch {epoch} - F1: {f1:.4f}")
        print(f"Epoch {epoch} - Confusion Matrix: {confusion_matrix}")


        if config['output_dir'] and utils.is_main_process():
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': config
            }
            torch.save(
                checkpoint, os.path.join(config['output_dir'], 'checkpoint.pth'.format(epoch))
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


    # Load the best model
    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    report_str, report_dict = final_test(model_without_ddp, data_loader_test, device=device)


    with open(os.path.join(config['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    utils.visualize_report(report_dict, config['output_dir'])






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    parser.add_argument('--config', type=str, default='P4T_BAD2/5', help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)