import os.path
import random

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_range(data):
    x_min, x_max = np.min(data[:, :, 0]), np.max(data[:, :, 0])
    y_min, y_max = np.min(data[:, :, 1]), np.max(data[:, :, 1])
    z_min, z_max = np.min(data[:, :, 2]), np.max(data[:, :, 2])
    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


def plot_point_cloud(bad1, bad2, itop, title):
    pc1 = bad1[700, :, :]
    pc2 = bad2[442, :, :]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pc1[:, 0], -pc1[:, 1], pc1[:, 2], s=2, c=pc1[:, 2], cmap='viridis', alpha=1)
    ax.scatter(itop[:, 0], -itop[:, 1], itop[:, 2], s=2, c=itop[:, 2], cmap='viridis', alpha=1)
    ax.scatter(pc2[:, 0], -pc2[:, 1], pc2[:, 2], s=2, c=pc2[:, 2], cmap='viridis', alpha=1)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    # ax.set_ylim([-2, 2])

    ax.view_init(elev=0, azim=0)

    plt.show()


def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as file:
        data = file['data'][:]
        print(data.shape)
    return data


if __name__ == "__main__":
    # Load data from H5 files
    bad1 = load_data_from_h5(os.path.join(os.getcwd(), '..\\BAD\\S1_file_5_point_clouds.h5'))
    bad2 = load_data_from_h5(os.path.join(os.getcwd(), '..\\BAD\\S3_file_80_point_clouds.h5'))
    itop_path = os.path.join(os.getcwd(), '..\\..\\..\\..\\ITOP_side_test_point_cloud_preprocessed')
    npz_files = [f for f in os.listdir(itop_path) if f.endswith('.npz')]
    chosen_file = random.choice(npz_files)
    file_path = os.path.join(itop_path, chosen_file)
    data = np.load(file_path)
    keys = data.keys()
    itop_point_cloud = data['arr_0']
    print(itop_point_cloud.shape)

    # Get the range of x, y, z values for each dataset
    # range1 = get_range(data1)
    # range2 = get_range(data2)
    # print(f'BAD ranges:\nX: {range1[0]}\nY: {range1[1]}\nZ: {range1[2]}')
    # print(f'ITOP SIDE ranges:\nX: {range2[0]}\nY: {range2[1]}\nZ: {range2[2]}')

    plot_point_cloud(bad1, bad2, itop_point_cloud,'BAD and ITOP Comparison')
