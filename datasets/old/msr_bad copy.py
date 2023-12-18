import os
import sys
import numpy as np
from augmentations import rotating_clip, mirror_x_axis_clip, mirror_y_axis_clip, random_jitter
from utils import read_csv, get_video_split
from torch.utils.data import Dataset

class BAD(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, max_frame_interval=20, num_points=2048, train=True, split_file='/data/iballester/datasets/BAD/sets.txt', augmentation_vector=None):
        super(BAD, self).__init__()
        
        self.max_frame_interval = max_frame_interval
        self.videos, self.labels, self.index_map = [], [], []
        index = 0
        self.split_info = read_csv(split_file)
        self.augmentation_vector = augmentation_vector

        for video_name in os.listdir(root):
            if video_name.startswith("."):
                continue

            split = get_video_split(video_name,self.split_info)

            label = int(video_name.split('_')[3])-1

            if label == 0 or label == 9:
                 continue

            if (split=="train") and train:
            
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

                self.videos.append(video)
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

            if (split=="val" or split=="test") and not train:

                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

                self.videos.append(video)
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
        
        if self.train:

            while True:
                varying_frame_interval = np.random.randint(self.frame_interval, self.max_frame_interval)
                if t + self.frames_per_clip * varying_frame_interval <= len(video):
                    break

            clip = [video[t+i*varying_frame_interval] for i in range(self.frames_per_clip)]
            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
            clip = np.array(clip)



             # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

            angle = np.random.uniform(0, 360)
            clip = rotating_clip(clip, angle)
            clip = np.array(clip)

            prob_mirror_x = np.random.rand()
            if prob_mirror_x > 0.5:
                clip = mirror_x_axis_clip(clip)

            clip = np.array(clip)


            prob_mirror_y = np.random.rand()
            if prob_mirror_y > 0.5:
                clip = mirror_y_axis_clip(clip)

            clip = np.array(clip)

            clip = random_jitter(clip)

            clip = np.array(clip)  

        else:
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

        clip = clip / 300
        return clip.astype(np.float32), label, index

if __name__ == '__main__':
    dataset = BAD(root='/data/iballester/datasets/BAD/f_depth_npz', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)