import os
import numpy as np

folder_path = "/data/iballester/datasets/BAD/f_depth_npz"
frame_statistics = {}

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".npz"):
        # Parse the file name to extract relevant information
        parts = filename.split("_")
        person = parts[0]
        class_num = int(parts[3])
        start_frame = int(parts[4])
        end_frame = int(parts[5].split(".")[0])
        
        # Calculate the number of frames for this file
        num_frames = end_frame - start_frame + 1
        
        # Update the statistics
        if person not in frame_statistics:
            frame_statistics[person] = {}
        if class_num not in frame_statistics[person]:
            frame_statistics[person][class_num] = 0
        frame_statistics[person][class_num] += num_frames

# Sort the person names and class numbers
sorted_persons = sorted(frame_statistics.keys())
sorted_classes = sorted(set(class_num for person_stats in frame_statistics.values() for class_num in person_stats.keys()))

# Print the statistics in order
for person in sorted_persons:
    print(f"Person: {person}")
    for class_num in sorted_classes:
        if class_num in frame_statistics[person]:
            num_frames = frame_statistics[person][class_num]
            print(f"Class {class_num}: {num_frames} frames")
    print()

# Calculate total statistics
total_statistics = {}
for person_stats in frame_statistics.values():
    for class_num, num_frames in person_stats.items():
        if class_num not in total_statistics:
            total_statistics[class_num] = 0
        total_statistics[class_num] += num_frames

print("Total statistics:")
for class_num in sorted_classes:
    if class_num in total_statistics:
        num_frames = total_statistics[class_num]
        print(f"Class {class_num}: {num_frames} frames")