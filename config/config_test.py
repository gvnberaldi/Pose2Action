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

    if bad_train == 'BAD1':
        data_train_path = '/data/iballester/datasets/BAD1/f_depth_npz_0-8_7'
        split_train_path = '/data/iballester/datasets/BAD1/sets_0-8_7.txt'
    else:
        data_train_path = '/data/iballester/datasets/BAD2/f_depth_npz_0-8_7'
        split_train_path = '/data/iballester/datasets/BAD2/sets_0-8_7.txt'

    if bad_test == 'BAD1':
        data_test_path = '/data/iballester/datasets/BAD1/f_depth_npz_0-8_7'
        split_test_path = '/data/iballester/datasets/BAD1/sets_0-8_7.txt'
    else:
        data_test_path = '/data/iballester/datasets/BAD2/f_depth_npz_0-8_7'
        split_test_path = '/data/iballester/datasets/BAD2/sets_0-8_7.txt'

    config['data_train_path'] = data_train_path
    config['split_train_path'] = split_train_path
    config['data_test_path'] = data_test_path
    config['split_test_path'] = split_test_path

    return config

