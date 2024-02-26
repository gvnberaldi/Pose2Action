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
        
    if 'mode_train' in config:
        print(config['mode_train'])
    else:
        print("Key 'mode_train' does not exist in the config.")

    if config['dataset'] == 'MSRA':
        config['dataset_path'] = path.MSRACTION_PATH

    elif config['dataset'] == 'Synthia':
        config['dataset_root'] = path.SYNTHIA_PATH
        config['dataset_path'] = config['dataset_root'] + '/processed_pc'
        config['data_eval'] = config['dataset_root'] + '/trainval_raw.txt'
        config['data_train'] = config['dataset_root'] + '/test_raw.txt'
        config['label_weights'] = config['dataset_root'] + '/labelweights.npz'

    elif config['dataset'] == 'BAD2':
        config['dataset_root'] = path.BAD2_PATH
        config['data_train_path'] = config['dataset_root'] + '/f_depth_npz_0-8_7'
        config['data_test_path'] = config['dataset_root'] + '/f_depth_npz_0-8_7'
        config['split_train_path'] = config['dataset_root'] + config['split']
        config['split_test_path'] = config['dataset_root'] + config['split']

    elif config['dataset'] == 'NTU60':
        config['dataset_root'] = path.NTU60_PATH
        config['dataset_path'] = config['dataset_root'] + '/pc'

        config['data_meta'] = config['dataset_root'] + '/ntu60.list'

    
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")
    
    
    if config['log_dir']:
        config['output_dir'] = os.path.join(path.EXPERIMENTS_PATH, config_file, config['log_dir'])
        print(f"Output dir: {config['output_dir']}")
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        else:
            if os.listdir(config['output_dir']) and config['mode_train']:
                raise FileExistsError(f"Directory {config['output_dir']} already exists and is not empty")

    return config

def load_config_hyper(config_file):

    path_config_file = os.path.join(path.HYPER_EXPERIMENTS_PATH, config_file, 'config.yaml')
    with open(path_config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            # Convert specific values to floats
            config['momentum'] = float(config['momentum'])
            config['weight_decay'] = float(config['weight_decay'])
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    if config['dataset'] == 'BAD2':
        config['dataset_root'] = path.BAD2_PATH
        config['data_train_path'] = config['dataset_root'] + '/f_depth_npz_0-8_7'
        config['data_test_path'] = config['dataset_root'] + '/f_depth_npz_0-8_7'
        config['split_train_path'] = config['dataset_root'] + config['split']
        config['split_test_path'] = config['dataset_root'] + config['split']

    
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")
    
    
    if config['log_dir']:
        config['output_dir'] = os.path.join(path.HYPER_EXPERIMENTS_PATH, config_file, config['log_dir'])
        print(f"Output dir: {config['output_dir']}")
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        else:
            if os.listdir(config['output_dir']):
                raise FileExistsError(f"Directory {config['output_dir']} already exists and is not empty")

    return config
