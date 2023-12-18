import numpy as np

clip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
num_slices=3
sliced_frames = np.array_split(clip, num_slices)

print(sliced_frames)
index_mapping = np.random.permutation(num_slices)
print(index_mapping)

 # Shuffle the sets based on the shuffled index mapping
shuffled_frames = [sliced_frames[i] for i in index_mapping]
print(shuffled_frames)


# Concatenate the shuffled sets
shuffled_clip = np.concatenate(shuffled_frames, axis=0)

print(shuffled_clip)