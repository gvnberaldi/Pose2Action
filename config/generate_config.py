import yaml
import sys
import const.path as path
import os

def load_yaml_file(file_path):
    """Load a YAML file and return its content."""
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

def convert_to_float(config, keys):
    """Convert specific values in the config to floats."""
    for key in keys:
        config[key] = float(config[key])

def check_mode_train(config):
    """Check if 'mode_train' exists in the config and print its value."""
    print(config.get('mode_train', "Key 'mode_train' does not exist in the config."))

def set_dataset_paths(config):
    """Set the dataset paths in the config based on the dataset name."""
    if config['dataset'] == 'ITOP-SIDE':
        config['dataset_root'] = path.ITOP_SIDE_PATH
        config['dataset_path'] = path.ITOP_SIDE_PATH
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")

def set_output_dir(config, config_file):
    """Set the output directory in the config and create it if it doesn't exist."""
    if config['log_dir']:
        config['output_dir'] = os.path.join(path.EXPERIMENTS_PATH, config_file, config['log_dir'])
        print(f"Output dir: {config['output_dir']}")
        os.makedirs(config['output_dir'], exist_ok=True)
        if os.listdir(config['output_dir']) and config['mode_train']:
            raise FileExistsError(f"Directory {config['output_dir']} already exists and is not empty")

def load_config(config_file):
    """Load a config file and return its content."""
    path_config_file = os.path.join(path.EXPERIMENTS_PATH, config_file, 'config.yaml')
    config = load_yaml_file(path_config_file)
    convert_to_float(config, ['momentum', 'weight_decay'])
    check_mode_train(config)
    set_dataset_paths(config)
    # set_output_dir(config, config_file)
    return config