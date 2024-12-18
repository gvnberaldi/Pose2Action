import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import shutil
import sys
import tempfile

def create_gif(npz_path, output_dir):
    # Check if the input file exists
    if not os.path.exists(npz_path):
        print(f"Input file {npz_path} does not exist.")
        return

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return

    try:
        # Load the .npz file
        data = np.load(npz_path, allow_pickle=True)

        # Extract the sequence of point clouds
        point_clouds = data['point_clouds']

        print(f"point_clouds shape: {point_clouds.shape}")
        

        # Initialize a list to store the frames
        frames = []

        frames_dir = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', ''))
        os.makedirs(frames_dir, exist_ok=True)

        # Iterate over the point clouds
        for i, pc in enumerate(point_clouds):
            print(pc.shape)
            print(f"iterating over point cloud {i} of {len(point_clouds)}")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=pc[:, 2], cmap='viridis')

            # Set the limits of the axes
            #ax.set_xlim([-0.5, 0.5])
            #ax.set_ylim([-0.4, 0.6])
            #ax.set_zlim([1.20, 2.1])


            #ax.view_init(elev=90, azim=-90)  # side view


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
        gif_name = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '.gif'))
        imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)

    except Exception as e:
        print(f"An error occurred: {e}")
    #finally:
        # Delete the temporary directory
        #shutil.rmtree(temp_dir)


if __name__ == "__main__":
    npz_path = "C:\\Users\\Utente\\Desktop\\thesis temp files\\S1_file_5_1_300_559.npz"
    output_dir = "C:\\Users\\Utente\\Desktop\\thesis temp files"
    create_gif(npz_path, output_dir)

