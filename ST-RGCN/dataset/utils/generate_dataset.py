import pickle
import struct
from typing import Tuple, List

import h5py
import numpy as np
from pathlib import Path

import yaml
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
import imageio.v2 as imageio
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


# Helper function to invert extrinsic transformation and return rotation and translation
def get_rotation_and_translation(extrinsic: List) -> Tuple[np.ndarray, np.ndarray]:
    inv_extrinsic = np.linalg.inv(extrinsic)
    rotation = inv_extrinsic[:3, :3]
    translation = inv_extrinsic[:3, 3]

    # Create a rotation object for a rotation of 55 degrees around the Z-axis
    theta_degrees = 55
    theta_radians = np.radians(theta_degrees)
    z_rotation = R.from_euler('z', theta_radians)
    new_rotation = z_rotation.as_matrix()
    rotation = rotation @ new_rotation

    return rotation, translation


# Read skeleton data from HDF5 file
def read_hdf5(filename: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with h5py.File(filename, 'r') as f:
            skeleton_data = f['joints_coordinates'][:]
            visible_joints = f['visible_joints'][:]
            outside_joints = f['outside_joints'][:]
            print(f'2D Skeleton data shape: {skeleton_data.shape}')
            print(f'Joints visibility data shape: {visible_joints.shape}')
            print(f'Joints outside data shape: {outside_joints.shape}')
            return skeleton_data, visible_joints, outside_joints
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return np.array([]), np.array([]), np.array([])


# Function to read depth data from a binary file
def read_bin(filename: Path) -> np.ndarray:
    try:
        with filename.open('rb') as f:
            num_frames, width, height = struct.unpack("<LLL", f.read(12))
            depth_data = f.read(num_frames * height * width * 4)
        depth = np.frombuffer(depth_data, dtype=np.uint32)
        depth = np.reshape(depth, (num_frames, height, width))
        print(f'Depth map data shape: {depth.shape}')
        return depth
    except Exception as e:
        print(f"Error reading depth map file: {e}")
        return np.array([])


def create_depth_map_array(folder_path):
    # Read the first image to get the dimensions
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Extract the number after 'f_img_' and before '.png' and sort

    first_image = imageio.imread(os.path.join(folder_path, files[0]))
    height, width = first_image.shape

    # Initialize an empty array to store all frames
    num_frames = len(files)
    depth_map_array = np.zeros((num_frames, height, width))

    # Read each image and add to the array
    for i, file_name in tqdm(enumerate(files), total=num_frames, desc='Creating depth map'):
        img_path = os.path.join(folder_path, file_name)
        depth_map_array[i] = imageio.imread(img_path)

    return depth_map_array


# Function to generate an id for each frame
def generate_frame_ids(subject_id, frame_range):
    # Generate frame IDs
    frame_ids = [f"{subject_id}_{idx:05d}" for idx in range(frame_range[0], frame_range[1])]
    return frame_ids


# Function to save the 3d skeleton in the dataset
def save_3d_skeleton(skeleton_3d: np.ndarray, ids:list, filename: Path):
    try:
        with h5py.File(filename, 'a') as f:
            # Check if the dataset already exists
            if '3d_joints_coordinates' in f:
                del f['3d_joints_coordinates']  # Delete it if it exists
            if 'id' in f:
                del f['id']  # Delete it if it exists
            # Create new datasets
            f.create_dataset('3d_joints_coordinates', data=skeleton_3d)
            f.create_dataset('id', data=ids)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")


# Function to compute relative positions based on the standard skeleton
def compute_relative_positions(standard_skeleton, skeleton_edges, depth_map, focal, reference_frame):

    num_joints = standard_skeleton.shape[1]
    skeleton_3d = np.empty((1, num_joints, 3))
    frame_depth = maximum_filter(depth_map[reference_frame], size=5)
    skeleton_frame = standard_skeleton[0]

    # Convert 2D skeleton to 3D using the depth map
    nearest_z = None
    for joint_index, (x_2d, y_2d) in enumerate(skeleton_frame):
        # Adjusts the point (x, y) to ensure it lies within the bounds of the depth map
        x_2d = min(max(x_2d, 0), frame_depth.shape[1] - 1)
        y_2d = min(max(y_2d, 0), frame_depth.shape[0] - 1)
        z_3d = frame_depth[int(y_2d), int(x_2d)]
        if z_3d == 0 and nearest_z is not None:
            z_3d = nearest_z
        elif z_3d != 0:
            nearest_z = z_3d

        # Calculate 3D coordinates
        x_3d = (x_2d - frame_depth.shape[1] / 2) * z_3d / focal
        y_3d = (y_2d - frame_depth.shape[0] / 2) * z_3d / focal

        # Apply transformation and scale
        point = np.array([x_3d, y_3d, z_3d])
        rotated_point = rotate_points(point)
        # Center the skeleton around the point cloud mean
        skeleton_3d[0][joint_index] = rotated_point

    reference_skeleton_3d = skeleton_3d[0]
    relative_positions = {}

    for edge in skeleton_edges:
        joint_1, joint_2 = edge
        # Compute the relative position vector from joint_1 to joint_2
        relative_vector = reference_skeleton_3d[joint_2] - reference_skeleton_3d[joint_1]
        relative_positions[edge] = relative_vector

    # Store the dictionary in a binary file
    with open(os.path.join(os.getcwd(), '..\\standard_skeleton\\relative_position.pkl'), 'wb') as f:
        pickle.dump(relative_positions, f)

    return relative_positions


def rotate_points(points, scale=55, x_theta=-45, z_theta=180):
    # Create transformation matrix for rotation around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(x_theta)), -np.sin(np.radians(x_theta))],
                   [0, np.sin(np.radians(x_theta)), np.cos(np.radians(x_theta))]])

    # Apply rotation around the x-axis
    rotated_points = np.dot(points, Rx.T)

    # Create transformation matrix for rotation around the z-axis
    Rz = np.array([[np.cos(np.radians(z_theta)), -np.sin(np.radians(z_theta)), 0],
                   [np.sin(np.radians(z_theta)), np.cos(np.radians(z_theta)), 0],
                   [0, 0, 1]])

    # Apply rotation around the z-axis
    rotated_points = np.dot(rotated_points, Rz.T)

    rotated_points /= scale
    return rotated_points


# Function to convert 2D skeleton coordinates to 3D with depth information
def generate_3d_skeleton_and_point_clouds(skeleton_2d: np.ndarray, skeleton_edges: List, outside_joints: np.ndarray,
                                          standard_relative_positions: dict, depth_map: np.ndarray, focal: float,
                                          frame_range: Tuple[int, int], distance_threshold: float = 0.2) -> Tuple[np.ndarray, List[np.ndarray]]:

    start_frame, end_frame = frame_range

    num_frames = end_frame - start_frame
    num_joints = skeleton_2d.shape[1]
    skeleton_3d = np.empty((num_frames, num_joints, 3))
    point_clouds = []

    xx, yy = np.meshgrid(np.arange(depth_map.shape[2]), np.arange(depth_map.shape[1]))

    for counter, frame_index in tqdm(enumerate(range(start_frame, end_frame)),
                                     total=num_frames, desc='Iterating over frames'):
        frame_depth = maximum_filter(depth_map[frame_index], size=5)
        skeleton_frame = skeleton_2d[frame_index]
        point_cloud_mean = [0., 0., 0.]
        cx = frame_depth.shape[1] / 2
        cy = frame_depth.shape[0] / 2
        # Create point cloud from depth map
        x = xx[frame_depth > 0]
        y = yy[frame_depth > 0]
        z = frame_depth[frame_depth > 0]
        if z.size > 0:
            x = (x - cx) * z / focal
            y = (y - cy) * z / focal
            points = np.stack([x, y, z], axis=0)
            point_cloud = rotate_points(points.T)
            point_cloud_mean = np.mean(point_cloud, axis=0)
            point_cloud -= point_cloud_mean

        # Convert 2D skeleton to 3D using the depth map
        skeleton_frame_3d = np.empty((num_joints, 3))
        nearest_z = None
        for joint_index, (x_2d, y_2d) in enumerate(skeleton_frame):
            # Do not compute the coordinate of the joint if it's not visible
            if outside_joints[frame_index][joint_index] == 1:
                continue

            # Adjusts the point (x, y) to ensure it lies within the bounds of the depth map
            x_2d = min(max(x_2d, 0), frame_depth.shape[1] - 1)
            y_2d = min(max(y_2d, 0), frame_depth.shape[0] - 1)

            z_3d = frame_depth[int(y_2d), int(x_2d)]
            if z_3d == 0 and nearest_z is not None:
                z_3d = nearest_z
            elif z_3d != 0:
                nearest_z = z_3d

            # Calculate 3D coordinates
            x_3d = (x_2d - cx) * z_3d / focal
            y_3d = (y_2d - cy) * z_3d / focal

            # Apply transformation and scale
            point = np.array([x_3d, y_3d, z_3d])
            rotated_point = rotate_points(point)
            # Center the skeleton around the point cloud mean
            rotated_point -= point_cloud_mean
            skeleton_frame_3d[joint_index] = rotated_point

        # Infer the relative standard position of the outside joint
        for joint_1, joint_2 in skeleton_edges:
            if (outside_joints[frame_index][joint_1] == 0) and (outside_joints[frame_index][joint_2] == 1):
                # Calculate the position of joint_2 based on joint_1 and the relative vector
                start = skeleton_frame_3d[joint_1]
                relative_vector = standard_relative_positions.get((joint_1, joint_2), None)
                if relative_vector is not None:
                    skeleton_frame_3d[joint_2] = start + relative_vector
                    outside_joints[frame_index][joint_2] = 0

            elif (outside_joints[frame_index][joint_2] == 0) and (outside_joints[frame_index][joint_1] == 1):
                # Calculate the position of joint_1 based on joint_2 and the relative vector
                start = skeleton_frame_3d[joint_2]
                relative_vector = standard_relative_positions.get((joint_1, joint_2), None)
                if relative_vector is not None:
                    skeleton_frame_3d[joint_1] = start - relative_vector
                    outside_joints[frame_index][joint_1] = 0

            '''
            # Check the distance between the joints and adjust the joint positions if the distance is too high
            distance = np.linalg.norm(skeleton_frame_3d[joint_2] - skeleton_frame_3d[joint_1])
            relative_vector = standard_relative_positions.get((joint_1, joint_2), None)
            if relative_vector is not None:
                expected_distance = np.linalg.norm(relative_vector)
                if abs(distance - expected_distance) > 0.5:
                    skeleton_frame_3d[joint_2] = skeleton_frame_3d[joint_1] + relative_vector
            '''

        # Clean the point cloud by removing points too far from skeleton joints
        if z.size > 0:
            distances = np.linalg.norm(point_cloud[:, np.newaxis, :] - skeleton_frame_3d[np.newaxis, :, :],
                                       axis=2)
            min_distances = np.min(distances, axis=1)
            point_cloud = point_cloud[min_distances <= distance_threshold]
            point_clouds.append(point_cloud)

        skeleton_3d[counter] = skeleton_frame_3d
        
    return skeleton_3d, point_clouds


# Function to plot 3D skeleton and point cloud
def plot_3d_skeleton_and_point_cloud(skeleton_3d: np.ndarray, point_clouds: List[np.ndarray],
                                     output_dir: Path) -> List[str]:
    image_files = []
    skeleton_edges = [
        (0, 1), (1, 2), (1, 3), (1, 8), (2, 4), (3, 5),
        (4, 6), (5, 7), (8, 9), (8, 10), (9, 11), (10, 12),
        (11, 13), (12, 14),
    ]
    cmap = plt.get_cmap('tab20')

    for i, (skeleton_frame, point_cloud) in tqdm(enumerate(zip(skeleton_3d, point_clouds)),
                                                 total=len(point_clouds), desc="Iterating over point clouds"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')


        # Plot skeleton
        ax.scatter(skeleton_frame[:, 0], -skeleton_frame[:, 1], skeleton_frame[:, 2],
                   c=skeleton_frame[:, 2], marker='o', cmap='viridis')
        for c, edge in enumerate(skeleton_edges):
            start, end = edge
            color = cmap(c % cmap.N)  # Cycle through the colors in the colormap
            ax.plot(
                [skeleton_frame[start, 0], skeleton_frame[end, 0]],
                [-skeleton_frame[start, 1], -skeleton_frame[end, 1]],
                [skeleton_frame[start, 2], skeleton_frame[end, 2]],
                c=color, linewidth=4,
            )

        # Label the joints with text
        for joint_index, joint in enumerate(skeleton_frame):
            ax.text(
                joint[0] + 0.04, -joint[1] + 0.04, joint[2] + 0.04,
                str(joint_index),
                color='red', fontsize=10, ha='center', va='center'
            )

        # Plot point cloud
        ax.scatter(
            point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2],
            s=2, c=point_cloud[:, 2], cmap='viridis', alpha=0.1
        )

        # Label axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Adjust axis limits to make the point cloud smaller
        ax.set_xlim([-1, 1])
        # ax.set_ylim([2, 2])
        ax.set_zlim([-2, 2])

        # Adjust viewpoint
        ax.view_init(elev=90, azim=90)

        # Remove the axes for a cleaner look
        # ax.axis('off')

        image_filename = os.path.join(output_dir, f'plot_{i + 1}.png')
        plt.savefig(image_filename)
        image_files.append(image_filename)
        plt.close(fig)

    return image_files


# Function to plot standard skeleton
def plot_standard_skeleton(relative_positions):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Initialize points
    points = {0: np.array([0, 0, 0])}

    # Plot initial point
    ax.scatter(0, 0, 0, c=0, marker='o', cmap='viridis', s=50, alpha=0.5)
    # ax.text(0.07, 0.07, 0.07, '0', color='black', fontsize=12, ha='center', va='center')

    cmap = plt.get_cmap('tab20')
    # Plot remaining points and edges
    for c, ((p1, p2), rel_pos) in enumerate(relative_positions.items()):
        color = cmap(c % cmap.N)
        if p2 not in points:
            points[p2] = points[p1] + rel_pos
            ax.scatter(points[p2][0], -points[p2][1], points[p2][2], c=points[p2][2], marker='o', cmap='viridis', s=50, alpha=0.5)
            # ax.text(points[p2][0] + 0.07, -points[p2][1] + 0.07, points[p2][2] + 0.07, str(p2),
                    # color='black', fontsize=12, ha='center', va='center')
        ax.plot([points[p1][0], points[p2][0]],
                [-points[p1][1], -points[p2][1]],
                [points[p1][2], points[p2][2]],
                color=color, linewidth=5, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Adjust axis limits to make the point cloud smaller
    ax.set_xlim([-0.9, 0.5])
    ax.set_ylim([0, 1.8])
    ax.set_zlim([-2, 2])

    # Adjust viewpoint
    # ax.view_init(elev=90, azim=90)

    # Remove the axes for a cleaner look
    # ax.axis('off')

    ax.set_title('Standard Skeleton Position')

    plt.savefig('C:\\Users\\Utente\\Documents\\University\\Magistrale\\Secondo Anno - 2023 - 2024\\Secondo Semetre - ERASMUS TU Wien\\Project in Visual Computing\\images\\strandard_skeleton_1.png', dpi=300, bbox_inches='tight')
    plt.show()


def reshape_point_cloud(point_clouds, num_points, frame_range):
    reshaped_point_clouds = []
    deleted_point_cloud = []
    start, end = frame_range
    for idx, point_cloud in enumerate(point_clouds):
        if point_cloud.shape[0] == 0:
            deleted_point_cloud.append(idx)
            start += 1
            continue
        if point_cloud.shape[0] > num_points:
            r = np.random.choice(point_cloud.shape[0], size=num_points, replace=False)
        elif point_cloud.shape[0] < num_points:
            repeat, residue = divmod(num_points, point_cloud.shape[0])
            r = np.concatenate([np.arange(point_cloud.shape[0])] * repeat + [np.random.choice(point_cloud.shape[0], size=residue, replace=False)], axis=0)
        else:
            reshaped_point_clouds.append(point_cloud)
        reshaped_point_clouds.append(point_cloud[r, :])
    new_frame_range = (start, end)
    return reshaped_point_clouds, deleted_point_cloud, new_frame_range


# Function to create a GIF from a list of images
def create_gif(image_files: List[str], output_path: Path, duration: float):
    images = [imageio.imread(img) for img in image_files]
    imageio.mimsave(os.path.join(output_path, 'gif.gif'), images, duration=duration)


def get_details_by_id(file_name):
    # Load the YAML configuration file
    with open(os.path.join(os.getcwd(), '..\\config\\config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    details = config.get(file_name)
    if details is None:
        raise ValueError(f"{file_name} does not exist in config.yaml")
    return details


if __name__ == "__main__":

    focal = 555

    skeleton_edges = [
        (0, 1), (1, 2), (1, 3), (1, 8), (2, 4), (3, 5),
        (4, 6), (5, 7), (8, 9), (8, 10), (9, 11), (10, 12),
        (11, 13), (12, 14),
    ]

    standard_relative_positions_path = os.path.join(os.getcwd(), '..\\standard_skeleton\\relative_position.pkl')
    if os.path.exists(standard_relative_positions_path):
        # Restore the relative position from the binary file
        print("Loading standard relative positions...")
        with open(standard_relative_positions_path, 'rb') as f:
            standard_relative_positions = pickle.load(f)
    else:
        reference_skeleton_depth_map_path = Path(os.path.join(os.getcwd(), '..\\standard_skeleton\\depth_map.bin'))
        reference_skeleton_annotation_path = Path(os.path.join(os.getcwd(), '..\\standard_skeleton\\annotations.h5'))
        # Load reference data
        reference_depth_map = read_bin(reference_skeleton_depth_map_path)
        reference_skeleton_2d, _, _ = read_hdf5(reference_skeleton_annotation_path)
        # Compute standard relative position between joints
        standard_relative_positions = compute_relative_positions(reference_skeleton_2d,
                                                                 skeleton_edges, reference_depth_map, focal,
                                                                 4)

    plot_standard_skeleton(standard_relative_positions)

    '''
    file_name = 'S3_file_80'
    config = get_details_by_id(file_name)

    dataset_path = os.path.join(os.getcwd(), '..')
    labels_file_path = Path(os.path.join(dataset_path, f'BAD\\{file_name}_labels.h5'))
    depth_map_file_path = Path(os.path.join(dataset_path, f'depth_map\\{file_name}'))
    results_path = Path(os.path.join(dataset_path, 'test'))
    point_cloud_file_path = Path(os.path.join(dataset_path, f'BAD\\{file_name}_point_clouds.h5'))
    # Load data
    skeleton_2d, visible_joints, outside_joints = read_hdf5(labels_file_path)
    # depth_map = read_bin(depth_map_file_path)
    depth_map = create_depth_map_array(depth_map_file_path)
    print(f'Depth map data shape: {depth_map.shape}')

    frame_range = config['frame_range']
    # Convert 2D skeleton to 3D and extract point clouds
    skeleton_3d, point_clouds = generate_3d_skeleton_and_point_clouds(
        skeleton_2d=skeleton_2d,
        skeleton_edges=skeleton_edges,
        standard_relative_positions=standard_relative_positions,
        outside_joints=outside_joints,
        depth_map=depth_map,
        focal=focal,
        frame_range=frame_range,
    )
    print(f'Skeleton 3D shape: {skeleton_3d.shape}')
    print(f'Numer point cloud frames: {len(point_clouds)}')

    num_points = 4096
    reshaped_point_clouds, deleted_point_clouds, frame_range = reshape_point_cloud(point_clouds, num_points=num_points,
                                                                                   frame_range=frame_range)
    skeleton_3d = np.delete(skeleton_3d, deleted_point_clouds, axis=0)
    
    # Generate frames ids
    frame_ids = generate_frame_ids(subject_id=config['id'], frame_range=frame_range)
    
    # Save the point clouds in the dataset
    print('Saving point cloud in h5 dataset...')
    point_clouds = np.array(reshaped_point_clouds, dtype='float16')
    with h5py.File(point_cloud_file_path, 'w') as f:
        # Create a dataset within the HDF5 file and save the point cloud data
        f.create_dataset('id', data=frame_ids)
        f.create_dataset('data', data=point_clouds)

    # Save the 3d skeleton in the dataset
    save_3d_skeleton(skeleton_3d, frame_ids, labels_file_path)

    # Plot 3D skeletons with point clouds and text labels
    image_files = plot_3d_skeleton_and_point_cloud(skeleton_3d, reshaped_point_clouds, results_path)

    # Create a GIF from the plot images
    # create_gif(image_files, results_path, duration=0.2)
    '''