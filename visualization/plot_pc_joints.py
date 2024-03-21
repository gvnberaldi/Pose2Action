import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def create_gif(point_clouds, joint_coords, video_id, output_directory):
    video_id_tuple = tuple(video_id.flatten())
    clip_index = f"{video_id_tuple[0]:01}_{video_id_tuple[1]:05}".encode('utf-8')
    print("Name: ", clip_index)

    # Convert input lists to numpy arrays for easier manipulation
    point_clouds = np.asarray(point_clouds)
    joint_coords = np.asarray(joint_coords)

    gif_frames = []
    frames_directory = os.path.join(output_directory, str(clip_index), 'frames')
    os.makedirs(frames_directory, exist_ok=True)

    # Loop over each point cloud
    for i, point_cloud in enumerate(point_clouds):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')

        # Overlay the joint coordinates on the scatter plot
        ax.scatter(joint_coords[i][:, 0], -joint_coords[i][:, 1], joint_coords[i][:, 2], c='r', s=20)

        # Set the view limits and labels
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([0.5, 3.5])
        ax.view_init(elev=90, azim=90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        frame_path = os.path.join(frames_directory, f'frame_{i}.png')
        plt.savefig(frame_path, dpi=300)
        gif_frames.append(imageio.imread(frame_path))

        plt.close()

    # Combine the frames into a gif and save it
    gif_path = os.path.join(output_directory, str(clip_index), f'{clip_index}.gif')
    imageio.mimsave(gif_path, gif_frames, 'GIF', duration=0.2)

def gif_gt_out_pc(point_clouds, joint_coords, joints_output, video_id, output_directory):

    video_id_tuple = tuple(video_id.flatten())
    clip_index = f"{video_id_tuple[0]:01}_{video_id_tuple[1]:05}".encode('utf-8')
    print("Name: ", clip_index)

    # Convert input lists to numpy arrays for easier manipulation
    point_clouds = np.asarray(point_clouds)
    joint_coords = np.asarray(joint_coords)
    joints_output = np.asarray(joints_output)

    gif_frames = []
    frames_directory = os.path.join(output_directory, str(clip_index), 'frames')
    os.makedirs(frames_directory, exist_ok=True)

    # Loop over each point cloud
    for i, point_cloud in enumerate(point_clouds):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')

        # Overlay the joint coordinates on the scatter plot
        ax.scatter(joint_coords[i][:, 0], -joint_coords[i][:, 1], joint_coords[i][:, 2], c='r', s=20)

        # Overlay the joint coordinates on the scatter plot
        ax.scatter(joints_output[i][:, 0], -joints_output[i][:, 1], joints_output[i][:, 2], c='b', s=20)

        # Set the view limits and labels
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([0.5, 3.5])
        ax.view_init(elev=90, azim=90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        frame_path = os.path.join(frames_directory, f'frame_{i}.png')
        plt.savefig(frame_path, dpi=300)
        gif_frames.append(imageio.imread(frame_path))

        plt.close()

    # Combine the frames into a gif and save it
    gif_path = os.path.join(output_directory, f'{clip_index}.gif')
    imageio.mimsave(gif_path, gif_frames, 'GIF', duration=0.2)