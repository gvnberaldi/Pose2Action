import h5py
import numpy as np
import os
def merge_files(filename1, filename2, output_filename, order_filename):
    # Open the first file
    with h5py.File(filename1, 'r') as f:
        data1 = {key: f[key][:] for key in f.keys()}
    # Add 'is_weak' key to data1 with value False
    data1['is_weak'] = np.full(data1[next(iter(data1))].shape[0], False)

    # Open the second file
    with h5py.File(filename2, 'r') as f:
        data2 = {key: f[key][:] for key in f.keys()}
    # Add 'is_weak' key to data2 with value True
    data2['is_weak'] = np.full(data2[next(iter(data2))].shape[0], True)
    data2['is_valid'][:len(data2['is_valid'])//1.33] = 1  # Set the first half to valid

    # Merge the data
    merged_data = {key: np.concatenate([data1[key], data2[key]], axis=0) for key in data2.keys()}

    # Open the order file
    with h5py.File(order_filename, 'r') as f:
        order_data = f['id'][:]

    # Create a dictionary to map 'id' values to their indices in the order data
    order_dict = {id: i for i, id in enumerate(order_data)}

    # Sort the merged data based on the order of 'id' values in the order data
    sorted_indices = np.argsort([order_dict[id] for id in merged_data['id']])
    for key in merged_data.keys():
        merged_data[key] = merged_data[key][sorted_indices]

    # Save the merged data
    with h5py.File(output_filename, 'w') as f:
        for key, value in merged_data.items():
            f.create_dataset(key, data=value)

# Use the function
merge_files('/data/iballester/datasets/ITOP-CLEAN/SIDE/valid_train_labels.h5', 
            '/caa/Homes01/iballester/dev-svr/3hpe_pc/experiments/0-predictions/37-baseline/predictions.h5',
            '/data/iballester/datasets/ITOP-CLEAN/SIDE/weakly_train_labels_37_14.h5',
            '/data/iballester/datasets/ITOP-CLEAN/SIDE/train_labels.h5'
)


def check_output(filename1, filename2, output_filename, order_filename):
    # Open the input files
    with h5py.File(filename1, 'r') as f:
        data1 = {key: f[key][:] for key in f.keys()}
    with h5py.File(filename2, 'r') as f:
        data2 = {key: f[key][:] for key in f.keys()}

    # Open the output file
    with h5py.File(output_filename, 'r') as f:
        output_data = {key: f[key][:] for key in f.keys()}

    # Open the order file
    with h5py.File(order_filename, 'r') as f:
        order_data = f['id'][:]

    # Check the dimensions and data types of the arrays
    for key in output_data.keys():
        print(f"Checking key: {key}")
        print(f"Shape of output data: {output_data[key].shape}")
        print(f"Data type of output data: {output_data[key].dtype}")
        if key in data1 and key in data2:
            print(f"Shape of input data: {np.concatenate([data1[key], data2[key]], axis=0).shape}")
            print(f"Data type of input data: {np.concatenate([data1[key], data2[key]], axis=0).dtype}")
        else:
            print(f"Key '{key}' not found in input data")

    # Check the 'id' values
    print("Checking 'id' values...")
    print(f"'id' values in output file: {output_data['id']}")
    print(f"'id' values in input files: {np.concatenate([data1['id'], data2['id']], axis=0)}")

    # Check the order of 'id' values
    print("Checking order of 'id' values...")
    print(f"Order of 'id' values in output file is correct: {np.array_equal(output_data['id'], order_data)}")



check_output('/data/iballester/datasets/ITOP-CLEAN/SIDE/valid_train_labels.h5', 
            '/caa/Homes01/iballester/dev-svr/3hpe_pc/experiments/0-predictions/37-baseline/predictions.h5',
            '/data/iballester/datasets/ITOP-CLEAN/SIDE/weakly_train_labels_37_14.h5',
            '/data/iballester/datasets/ITOP-CLEAN/SIDE/train_labels.h5'
)