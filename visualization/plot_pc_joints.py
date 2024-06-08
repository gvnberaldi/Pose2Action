import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

import const.skeleton_joints
from const import skeleton_joints
from tqdm import tqdm


def create_gif(point_clouds, joint_coords, video_id, output_directory, plot_lines=True, label_frame='middle'):
    video_id_tuple = tuple(video_id.flatten())
    clip_index = f"{video_id_tuple[0]:01}_{video_id_tuple[1]:05}".encode('utf-8')
    print("Name: ", clip_index)

    # Convert input lists to numpy arrays for easier manipulations
    point_clouds = np.asarray(point_clouds)
    joint_coords = np.asarray(joint_coords)

    # Find the index of the central frame
    central_frame_index = len(point_clouds) // 2

    if label_frame == 'last':
        label_frame_index = len(point_clouds) - 1  # Calculate the index of the last frame
    elif label_frame == 'middle':
        label_frame_index = central_frame_index

    print(joint_coords.shape)

    gif_frames = []
    frames_directory = os.path.join(output_directory, str(clip_index), 'frames')
    os.makedirs(frames_directory, exist_ok=True)

    # Loop over each point cloud
    for i, point_cloud in tqdm(enumerate(point_clouds), total=len(point_clouds), desc='Iterating over point clouds'):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')

        # Overlay the joint coordinates on the scatter plot only for the central frame
        if i == label_frame_index:

            print(joint_coords.shape)
            ax.scatter(joint_coords[: ,:, 0], -joint_coords[:, :, 1], joint_coords[:, :, 2], c='r', s=20)
            # Draw lines representing limbs
            if plot_lines:
                _plot_skeleton(ax, joint_coords)

        # Adjust axis limits to make the point cloud smaller
        ax.set_xlim([-0.8, 1])
        ax.set_ylim([-1, 0.8])
        ax.set_zlim([-0.9, 0.9])
        # ax.view_init(elev=90, azim=90)
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


def gif_gt_out_pc(point_clouds, joint_coords, joints_output, video_id, output_directory, plot_lines=True, label_frame='middle'):

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
    # Find the index of the central frame
    
    central_frame_index = len(point_clouds) // 2

    if label_frame == 'last':
        label_frame_index = len(point_clouds) - 1  # Calculate the index of the last frame
    elif label_frame == 'middle':
        label_frame_index = central_frame_index

    # Loop over each point cloud
    for i, point_cloud in tqdm(enumerate(point_clouds), total=len(point_clouds), desc='Iterating over point clouds'):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')

        # Overlay the joint coordinates on the scatter plot only for the central frame
        if i == label_frame_index:
            ax.scatter(joint_coords[:,:, 0], -joint_coords[:, :, 1], joint_coords[:, :, 2], c='r', s=20)
            ax.scatter(joints_output[:,:, 0], -joints_output[:,:, 1], joints_output[:,:, 2], c='b', s=20)
            # Draw lines representing limbs

            if plot_lines:
                _plot_skeleton(ax, joint_coords)
                _plot_skeleton(ax, joints_output)

        # Set the view limits and labels
        ax.set_xlim([-1, 1])
        ax.view_init(elev=90, azim=90)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Remove the axes for a cleaner look
        # ax.axis('off')

        frame_path = os.path.join(frames_directory, f'frame_{i}.png')
        plt.savefig(frame_path, dpi=300)
        gif_frames.append(imageio.imread(frame_path))

        plt.close()

    gif_path = os.path.join(output_directory, f'{clip_index}.gif')
    imageio.mimsave(gif_path, gif_frames, 'GIF', duration=0.2)


def plot_skeleton(joints, plot_2d=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    _plot_skeleton(ax, joints)

    if plot_2d:
        ax.view_init(-90, 90)
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
    plt.show()


def _plot_skeleton(ax_context, joints):

    for (idx1, idx2, colour) in const.skeleton_joints.joint_connections:
        p1 = joints[0, idx1]
        p2 = joints[0, idx2]
        x = [p1[0], p2[0]]
        y = [-p1[1], -p2[1]]
        z = [p1[2], p2[2]]

        ax_context.plot(x, y, z, color=colour, linewidth=4)


def _add_3d_axis_indicator(ax):
    # Create an inset axes with a 3D plot for the axis indicator
    # Adjust the first two values to position the inset closer to the main plot
    # These values might need to be adjusted based on the size and position of your main plot
    axis_indicator = plt.axes([0.3, 0.18, 0.1, 0.1], projection='3d')
    
    # Hide the axis planes, ticks, and labels
    axis_indicator.set_xticks([])
    axis_indicator.set_yticks([])
    axis_indicator.set_zticks([])
    axis_indicator.axis('off')
    
    # Draw the three axes lines
    axis_indicator.quiver(0, 0, 0, 1, 0, 0, length=0.5, normalize=True, color='r', arrow_length_ratio=0.1)
    axis_indicator.quiver(0, 0, 0, 0, 1, 0, length=0.5, normalize=True, color='g', arrow_length_ratio=0.1)
    axis_indicator.quiver(0, 0, 0, 0, 0, 1, length=0.5, normalize=True, color='b', arrow_length_ratio=0.1)


def clean_create_gif(point_clouds, joint_coords, video_id, output_directory, plot_lines=True, label_frame='middle'):
    video_id_tuple = tuple(video_id.flatten())
    clip_index = f"{video_id_tuple[0]:01}_{video_id_tuple[1]:05}".encode('utf-8')
    print("Name: ", clip_index)

    # Convert input lists to numpy arrays for easier manipulations
    point_clouds = np.asarray(point_clouds)
    joint_coords = np.asarray(joint_coords)

    # Find the index of the central frame
    central_frame_index = len(point_clouds) // 2

    if label_frame == 'last':
        label_frame_index = len(point_clouds) - 1  # Calculate the index of the last frame
    elif label_frame == 'middle':
        label_frame_index = central_frame_index

    print(joint_coords.shape)

    gif_frames = []
    frames_directory = os.path.join(output_directory, str(clip_index), 'frames')
    os.makedirs(frames_directory, exist_ok=True)

    # Loop over each point cloud
    for i, point_cloud in enumerate(point_clouds):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')
        ax.grid(False)
        ax.set_axis_off()
        _add_3d_axis_indicator(ax)


        # Overlay the joint coordinates on the scatter plot only for the central frame
        if i == label_frame_index:

            print(joint_coords.shape)
            ax.scatter(joint_coords[: ,:, 0], -joint_coords[:, :, 1], joint_coords[:, :, 2], c='r', s=20)
            # Draw lines representing limbs
            if plot_lines:
                _plot_skeleton(ax, joint_coords)

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

    
def clean_gif_gt_out_pc(point_clouds, joint_coords, joints_output, video_id, output_directory, plot_lines=True, label_frame='middle'):

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
    # Find the index of the central frame

    
    central_frame_index = len(point_clouds) // 2

    if label_frame == 'last':
        label_frame_index = len(point_clouds) - 1  # Calculate the index of the last frame
    elif label_frame == 'middle':
        label_frame_index = central_frame_index


    # Loop over each point cloud
    for i, point_cloud in enumerate(point_clouds):
        # Create a 3D scatter plot for each point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')
        ax.grid(False)
        ax.set_axis_off()
        _add_3d_axis_indicator(ax)


        # Overlay the joint coordinates on the scatter plot only for the central frame
        if i == label_frame_index:
            #ax.scatter(joint_coords[:,:, 0], -joint_coords[:, :, 1], joint_coords[:, :, 2], c='r', s=20)
            ax.scatter(joints_output[:,:, 0], -joints_output[:,:, 1], joints_output[:,:, 2], c='b', s=20)
            # Draw lines representing limbs

            if plot_lines:
                #_plot_skeleton(ax, joint_coords)
                _plot_skeleton(ax, joints_output)


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

    gif_path = os.path.join(output_directory, f'{clip_index}.gif')
    imageio.mimsave(gif_path, gif_frames, 'GIF', duration=0.2)


def clean_sep_gt_out_pc(point_clouds, joint_coords, joints_output, video_id, output_directory, plot_lines=True, label_frame='middle'):

    video_id_tuple = tuple(video_id.flatten())
    clip_index = f"{video_id_tuple[0]:01}_{video_id_tuple[1]:05}".encode('utf-8')
    print("Name: ", clip_index)

    # Convert input lists to numpy arrays for easier manipulation
    point_clouds = np.asarray(point_clouds)
    joint_coords = np.asarray(joint_coords)
    joints_output = np.asarray(joints_output)

    # Create directories for ground truth and output joints
    gt_frames_directory = os.path.join(output_directory, str(clip_index), 'gt_frames')
    out_frames_directory = os.path.join(output_directory, str(clip_index), 'out_frames')
    os.makedirs(gt_frames_directory, exist_ok=True)
    os.makedirs(out_frames_directory, exist_ok=True)

    # Find the index of the central frame
    central_frame_index = len(point_clouds) // 2
    if label_frame == 'last':
        label_frame_index = len(point_clouds) - 1  # Calculate the index of the last frame
    elif label_frame == 'middle':
        label_frame_index = central_frame_index

    # Loop over each point cloud for ground truth joints
    for i, point_cloud in enumerate(point_clouds):
        fig, ax = _create_3d_scatter_plot(point_cloud)
        if i == label_frame_index and plot_lines:
            ax.scatter(joint_coords[:,:, 0], -joint_coords[:, :, 1], joint_coords[:, :, 2], c='r', s=20)
            _plot_skeleton(ax, joint_coords)
        _save_frame(fig, ax, gt_frames_directory, i)

    # Loop over each point cloud for output joints
    for i, point_cloud in enumerate(point_clouds):
        fig, ax = _create_3d_scatter_plot(point_cloud)
        if i == label_frame_index and plot_lines:
            ax.scatter(joints_output[:,:, 0], -joints_output[:,:, 1], joints_output[:,:, 2], c='b', s=20)
            _plot_skeleton(ax, joints_output)
        _save_frame(fig, ax, out_frames_directory, i)


def _create_3d_scatter_plot(point_cloud):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2], s=0.1, c=point_cloud[:, 2], cmap='viridis')
    ax.grid(False)
    ax.set_axis_off()
    _add_3d_axis_indicator(ax)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([0.5, 3.5])
    ax.view_init(elev=90, azim=90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax


def _save_frame(fig, ax, frames_directory, i):
    frame_path = os.path.join(frames_directory, f'frame_{i}.png')
    plt.savefig(frame_path, dpi=300)
    plt.close(fig)