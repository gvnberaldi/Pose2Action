import yaml
import sys

import yaml
import sys

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            # Convert specific values to floats
            config['momentum'] = float(config['momentum'])
            config['weight_decay'] = float(config['weight_decay'])
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return config

def construct_data_paths(config):
    bad_train = config['bad_train']
    bad_test = config['bad_test']

    config['data_train_path'] = '/data/iballester/datasets/MSRAction3D_Output'
    config['data_test_path'] = '/data/iballester/datasets/MSRAction3D_Output'

    return config

