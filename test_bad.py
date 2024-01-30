from __future__ import print_function
import os
import torch
import argparse
from config.generate_config import load_config
import models.model_factory as model_factory
import scripts.utils as utils
from trainer_bad import load_data, create_criterion, final_test

def main(args):

    print("Start testing")
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_args'])
    device = torch.device(0)

    utils.set_random_seed(config['seed'])

    # Data loading code
    data_loader, data_loader_test, _, num_classes = load_data(config)

    model = model_factory.create_model(config, num_classes)

    model.to(device)

    criterion = create_criterion(config, data_loader, num_classes, device)

    model_without_ddp = model

    if config['resume']:
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model_state_dict = checkpoint['model']
        model_without_ddp.load_state_dict(model_state_dict, strict=True)  # strict=False allows for partial loadin
    


        f1, report_str = final_test(model_without_ddp, criterion, data_loader_test, device=device, output_dir=config['output_dir'])
        print("F1: ", f1)
        print(report_str)

        with open(os.path.join(config['output_dir'], 'classification_report.txt'), 'w') as f:
            f.write(report_str)
    else:
        print("No checkpoint found")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P4Transformer Model Testing')
    parser.add_argument('--config', type=str, default='P4T_BAD2/10-test', help='Path to the YAML config file')

    args = parser.parse_args()
    main(args)