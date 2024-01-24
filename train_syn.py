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


from trainer_syn import train_one_epoch, evaluate, load_data, create_optimizer_and_scheduler

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
    data_loader, data_loader_test = load_data(config)

    num_classes=12 #TODO: FIX THIS    

    model = model_factory.create_model(config, num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss(weight=torch.from_numpy(data_loader.dataset.labelweights).to(device), reduction='none')
    criterion_test = nn.CrossEntropyLoss(weight=torch.from_numpy(data_loader_test.dataset.labelweights).to(device), reduction='none')

    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

    model_without_ddp = model

    if config['resume']:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    mIoU = 0
    
    for epoch in range(config['start_epoch'], config['epochs']):
        loss = train_one_epoch(model, criterion_train, optimizer, lr_scheduler, data_loader, device, epoch)
        eval_accuracy, eval_mIoU, eval_loss  = evaluate(model, criterion_test, data_loader_test, device=device)

        wandb.log({"train_loss": loss, "eval_accuracy": eval_accuracy, "eval_mIoU": eval_mIoU, "eval_loss": eval_loss})
        

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
        
        if eval_mIoU > mIoU:
            mIoU = eval_mIoU
            if config['output_dir'] and utils.is_main_process():
                torch.save(
                    checkpoint, os.path.join(config['output_dir'], 'best_model.pth')
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    parser.add_argument('--config', type=str, default='P4T_Synthia/1', help='Path to the YAML config file')
    args = parser.parse_args()
    main(args)