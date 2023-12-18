import os
import sys
import numpy as np
from torch.utils.data import Dataset

class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, max_frame_interval=20, num_points=2048, train=True, split_file='/data/iballester/datasets/BAD/sets_0-8_7.txt', aug=[], num_slices=3):
        super(MSRAction3D, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        self.num_slices = num_slices
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300

        sliced_frames = np.array_split(clip, self.num_slices)

        # Generate an index mapping to keep track of the shuffling
        index_mapping = np.random.permutation(self.num_slices)

        # Shuffle the sets based on the shuffled index mapping
        shuffled_frames = [sliced_frames[i] for i in index_mapping]

        # Concatenate the shuffled sets
        shuffled_clip = np.concatenate(shuffled_frames, axis=0)

        # Convert to NumPy arrays
        shuffled_clip = np.array(shuffled_clip)

        return shuffled_clip.astype(np.float32), label, index, index_mapping

if __name__ == '__main__':
    dataset = MSRAction3D(root='/data/iballester/datasets/MSRAction3D_Output/', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
