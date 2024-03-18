import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def create_gif(pc, joints, idx, output_dir):
    point_clouds = np.asarray(pc)
    joints = np.asarray(joints)

    frames = []

    frames_dir = os.path.join(output_dir, str(idx), 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    for i, pc in enumerate(point_clouds):

        print(pc.shape)
        print(f"iterating over point cloud {i} of {len(point_clouds)}")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], -pc[:, 1], s=0.1, c=pc[:, 2], cmap='viridis')

        joint_coords = joints[i]

        ax.scatter(joint_coords[:, 0], joint_coords[:, 2], -joint_coords[:, 1],   c='r', s=20)

        # Set the limits of the axes
        ax.set_xlim([-1, 1.5])
        #ax.set_ylim([-0.4, 0.6])
        ax.set_zlim([-1, 1.5])


        ax.view_init(elev=180, azim=-90)  # frontal view


        # Remove the axes for a cleaner look
        #ax.axis('off')

        # Set the names of the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #ax.set_box_aspect([1,1,1])

        # Save the frame in the temporary directory with a higher resolution
        frame_path = os.path.join(frames_dir, f'frame_{i}.png')
        plt.savefig(frame_path, dpi=300)
        frames.append(imageio.imread(frame_path))
        plt.close()

    # Create a gif from the frames

    gif_name = os.path.join(output_dir, str(idx), str(idx)+'.gif')
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)