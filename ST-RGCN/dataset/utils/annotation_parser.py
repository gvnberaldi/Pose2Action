import os.path
import xml.etree.ElementTree as ET
import numpy as np
import h5py

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), '..\\annotations\\S3_file_80_annotations.xml')
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Determine the number of frames
    max_frame = 0
    for track in root.findall('track'):
        # Iterate through all 'skeleton' elements within each 'track'
        for skeleton in track.findall('skeleton'):
            frame = int(skeleton.get('frame'))  # Extract the frame value
            if frame > max_frame:
                max_frame = frame

    # Initialize the joints coordinates 3D matrix with zeros
    joints_coordinates = np.zeros((max_frame + 1, 15, 2))
    # Initialize the joints visibility 2D matrix with zeros
    visible_joints = np.zeros((max_frame + 1, 15), dtype=int)
    # Initialize the occlude joints 2D matrix with zeros
    outside_joints = np.ones((max_frame + 1, 15), dtype=int)

    # Populate coordinates and visibility matrix
    for track in root.findall('track'):
        for skeleton in track.findall('skeleton'):
            frame = int(skeleton.get('frame'))  # Get the frame

            for joint in skeleton.findall('points'):
                joint_label = int(joint.get('label'))

                occluded = int(joint.get('occluded')) # Get the occluded value
                outside = int(joint.get('outside'))  # Get the outside value
                joint_visible = 1 if occluded == 0 and outside == 0 else 0
                joint_outside = 0 if outside == 0 else 1
                joint_coordinates = [float(val) for val in joint.get('points').split(',')] # Get (x, y) coordinates

                joints_coordinates[frame, joint_label] = joint_coordinates
                visible_joints[frame, joint_label] = joint_visible
                outside_joints[frame, joint_label] = joint_outside

    hdf5_file_path = os.path.join(os.getcwd(), '..\\BAD\\S3_file_80_labels.h5')

    # Create and populate the HDF5 file
    with h5py.File(hdf5_file_path, 'w') as hdf:
        # Create datasets and store the matrices
        hdf.create_dataset('joints_coordinates', data=joints_coordinates, dtype='float64')
        hdf.create_dataset('visible_joints', data=visible_joints, dtype='int')
        hdf.create_dataset('outside_joints', data=outside_joints, dtype='int')


