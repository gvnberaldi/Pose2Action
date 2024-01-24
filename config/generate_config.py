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

    elif config['dataset'] == 'Synthia':
        config['dataset_root'] = path.SYNTHIA_PATH
        config['dataset_path'] = config['dataset_root'] + '/processed_pc'
        config['data_eval'] = config['dataset_root'] + '/trainval_raw.txt'
        config['data_train'] = config['dataset_root'] + '/test_raw.txt'
        config['label_weights'] = config['dataset_root'] + '/labelweights.npz'
    
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")
    
    
    if config['output_dir']:
        os.makedirs(config['output_dir'], exist_ok=True)

    return config
