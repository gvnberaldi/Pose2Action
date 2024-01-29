import os
import sys
import numpy as np
from torch.utils.data import Dataset
from augmentations.AugPipeline import AugPipeline
import random

DS_AUGMENTS_CFG = []

def read_csv(file_path):
    split_info = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                video_name = parts[0]
                split = parts[1]
                split_info.append({'video_name': video_name, 'split': split})
    return split_info

def get_video_split(video_name, split_info):
    for video_info in split_info:
        if video_info['video_name'] == video_name:
             return video_info['split']
    return None

class BAD(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, max_frame_interval=20, num_points=2048, train=True, split_file='/data/iballester/datasets/BAD/sets_0-8_7.txt', aug=DS_AUGMENTS_CFG):
        super(BAD, self).__init__()
        self.max_frame_interval = max_frame_interval
        self.videos, self.labels, self.index_map = [], [], []
        index = 0
        self.split_info = read_csv(split_file)
        self.aug = aug

        for video_name in os.listdir(root):
            if video_name.startswith("."):
                continue

            split = get_video_split(video_name,self.split_info)
            label = int(video_name.split('_')[3])

            if label == 10 or label ==13 or label == 12:
                continue
            
            if (split=="train") and train:

                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

                self.videos.append(video)
                self.labels.append(label)

                nframes = video.shape[0]

                # Loop for the regular frames
                for t in range(0, nframes, frame_interval):
                    self.index_map.append((index, t))
                index += 1

            if (split=="val" or split=="test") and not train:

                if label == 10 or label ==13 or label == 12:
                 continue

                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

                self.videos.append(video)
                self.labels.append(label)
                nframes = video.shape[0]

                # Loop for the regular frames
                for t in range(0, nframes, frame_interval):
                    self.index_map.append((index, t))
                index += 1
                
        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    
    def __len__(self):
        return len(self.index_map)

    def select_points(self, p, num_points):
        """Selects a subset of points from p."""
        if p.shape[0] > num_points:
            r = np.random.choice(p.shape[0], size=num_points, replace=False)
        else:
            repeat, residue = num_points // p.shape[0], num_points % p.shape[0]
            r = np.random.choice(p.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
        return p[r, :]

    def create_clip(self, video, t, frame_interval, frames_per_clip):
        """Creates a clip from the video."""
        return [video[min(t+i*frame_interval, len(video)-1)] for i in range(frames_per_clip)] #Padding

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]
            
        if self.train:
            # Create the clip
            clip = self.create_clip(video, t, self.frame_interval, self.frames_per_clip)
            # Select the points
            for i, p in enumerate(clip):
                clip[i] = self.select_points(p, self.num_points)
            clip = np.array(clip)

            # Create and apply the augmentation pipeline
            aug_pipeline = AugPipeline()
            aug_pipeline.create_pipeline(self.aug)
            aug_clip, par, tensors = aug_pipeline.augment(clip)
            clip = np.array(aug_clip)   

        else:
            # Create the clip
            clip = self.create_clip(video, t, self.frame_interval, self.frames_per_clip)

            # Select the points
            for i, p in enumerate(clip):
                clip[i] = self.select_points(p, self.num_points)
            clip = np.array(clip)

        clip = clip / 30
        return clip.astype(np.float32), label, index, index

if __name__ == '__main__':
    dataset = BAD(root='/data/iballester/datasets/BAD/f_depth_npz_0-8_7', frames_per_clip=24)
    clip, label, video_idx = dataset[0]
    #print(clip)
    #print(label)
    #print(video_idx)
    #print(dataset.num_classes)