import os
import numpy as np
import cv2
import struct

def create_bin_files(text_file_path, depth_maps_folder_path, output_folder):
    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)  # Total number of lines

    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) !=5:
            continue

        file_name, subject, cls, start_frame, end_frame = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4])

        depth_map_folder = f'{depth_maps_folder_path}/{subject}_{file_name}'
        bin_filename = f'{subject}_{file_name}_{cls}_{start_frame}_{end_frame}.bin'
        bin_filepath = os.path.join(output_folder, bin_filename)

        depth_maps = []
        for frame_num in range(start_frame, end_frame + 1):
            depth_map_filename = os.path.join(depth_map_folder, f'f_img_{frame_num}.png')
            depth_map = cv2.imread(depth_map_filename, cv2.IMREAD_UNCHANGED)
            depth_maps.append(depth_map)

        depth_maps = np.array(depth_maps)
        num_frames, height, width = depth_maps.shape
        depth_maps = depth_maps.astype(np.int32).flatten()

        with open(bin_filepath, 'wb') as bin_file:
            bin_file.write(struct.pack("<L", num_frames))
            bin_file.write(struct.pack("<L", width))
            bin_file.write(struct.pack("<L", height))
            bin_file.write(depth_maps.tobytes())

        print(f"Processed {i+1} out of {total_lines} lines")

if __name__ == "__main__":
    text_file_path = "/Volumes/Irene_CVL2/BAD-dataset/BAD2/activity_labels.txt"
    depth_maps_folder_path = "/Volumes/Irene_CVL2/BAD-dataset/BAD2/f_depth"
    output_folder = "/Volumes/Irene_CVL2/BAD-dataset/BAD2/f_depth_bin/"
    create_bin_files(text_file_path, depth_maps_folder_path, output_folder)