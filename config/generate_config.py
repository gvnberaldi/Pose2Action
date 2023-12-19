import yaml
import sys
import const.path as path
import os

def load_config(config_file):

    path_config_file = os.path.join(path.EXPERIMENTS_PATH, config_file, 'config.yaml')
    with open(path_config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            # Convert specific values to floats
            config['momentum'] = float(config['momentum'])
            config['weight_decay'] = float(config['weight_decay'])
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    if config['dataset'] == 'MSRA':
        config['dataset_path'] = path.MSRACTION_PATH
    
    if config['output_dir']:
        os.makedirs(config['output_dir'], exist_ok=True)

    return config
