import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


# Function to load and concatenate datasets from multiple HDF5 files
def load_and_concatenate_hdf5_files(folder_path):
    # List all HDF5 files in the folder with the specified prefix for labels and point clouds
    label_files = [f for f in os.listdir(folder_path) if f.endswith('labels.h5')]
    point_cloud_files = [f for f in os.listdir(folder_path) if f.endswith('point_clouds.h5')]

    # Initialize empty lists to store data from each file
    all_identifiers = []
    all_point_clouds = []
    all_joints = []

    # Loop through each HDF5 file and load the datasets
    for label_file, point_cloud_file in zip(label_files, point_cloud_files):
        label_file_path = os.path.join(folder_path, label_file)
        point_cloud_file_path = os.path.join(folder_path, point_cloud_file)

        # Open the label HDF5 file
        with h5py.File(label_file_path, 'r') as label_f:
            # Append the data to the lists
            all_identifiers.append(label_f['id'][:])
            all_joints.append(label_f['3d_joints_coordinates'][:])

        # Open the point cloud HDF5 file
        with h5py.File(point_cloud_file_path, 'r') as pc_f:
            # Append the point cloud data to the list
            all_point_clouds.append(pc_f['data'][:])

    # Concatenate the data from all files
    concatenated_identifiers = np.concatenate(all_identifiers, axis=0)
    concatenated_point_clouds = np.concatenate(all_point_clouds, axis=0)
    concatenated_joints = np.concatenate(all_joints, axis=0)

    return concatenated_identifiers, concatenated_point_clouds, concatenated_joints


# Function to split the array into blocks
def split_into_blocks(array, block_size):
    return [array[i:i + block_size] for i in range(0, len(array), block_size)]


if __name__ == "__main__":
    '''
    identifiers, point_clouds, joints = load_and_concatenate_hdf5_files(os.path.join(os.getcwd(), "..\\BAD"))
   
    block_size = 15
    # Split the datasets into blocks
    identifier_blocks = split_into_blocks(identifiers, block_size)
    point_cloud_blocks = split_into_blocks(point_clouds, block_size)
    joint_blocks = split_into_blocks(joints, block_size)

    # Zip the blocks together
    blocks = list(zip(identifier_blocks, point_cloud_blocks, joint_blocks))

    # Shuffle the blocks
    np.random.shuffle(blocks)

    # Recombine the shuffled blocks into a single dataset
    shuffled_identifiers = np.concatenate([block[0] for block in blocks], axis=0)
    shuffled_point_clouds = np.concatenate([block[1] for block in blocks], axis=0)
    shuffled_joints = np.concatenate([block[2] for block in blocks], axis=0)

    # Split the shuffled dataset into training and testing sets
    train_ratio = 0.8  # Example ratio
    train_size = int(len(shuffled_identifiers) * train_ratio)

    train_identifiers = shuffled_identifiers[:train_size]
    test_identifiers = shuffled_identifiers[train_size:]

    train_point_clouds = shuffled_point_clouds[:train_size]
    test_point_clouds = shuffled_point_clouds[train_size:]

    train_joints = shuffled_joints[:train_size]
    test_joints = shuffled_joints[train_size:]

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\train_point_clouds.h5"), 'w') as f:
        f.create_dataset('id', data=train_identifiers)
        f.create_dataset('data', data=train_point_clouds)

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\train_labels.h5"), 'w') as f:
        f.create_dataset('id', data=train_identifiers)
        f.create_dataset('3d_joints_coordinates', data=train_joints)

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\test_point_clouds.h5"), 'w') as f:
        f.create_dataset('id', data=test_identifiers)
        f.create_dataset('data', data=test_point_clouds)

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\test_labels.h5"), 'w') as f:
        f.create_dataset('id', data=test_identifiers)
        f.create_dataset('3d_joints_coordinates', data=test_joints)
    '''

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\train_labels.h5"), 'r') as file:
        train_ids = file['id'][:]

    with h5py.File(os.path.join(os.getcwd(), "..\\BAD\\final_dataset\\test_labels.h5"), 'r') as file:
        test_ids = file['id'][:]

    shared_ids = []
    print(len(train_ids))
    for train_id in train_ids:
        if train_id in test_ids:
            shared_ids.append(train_id)
    print(len(shared_ids))
    if shared_ids:
        print(f"Shared IDs: {shared_ids}")
    else:
        print("No shared IDs found.")

