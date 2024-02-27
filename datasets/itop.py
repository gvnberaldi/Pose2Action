import os
import sys
import numpy as np
import h5py
import collections

from torch.utils.data import Dataset

import time


class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, subset_size=5000):
        super(ITOP, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []

        # LOAD DATA FROM FILES
        point_clouds_file = ''
        labels_file = ''

        if train:
            point_clouds_file = "train_point_cloud.h5"
            labels_file = "train_labels.h5"
        if not train:
            point_clouds_file = "test_point_cloud.h5"
            labels_file = "test_labels.h5"

        start_time = time.time()

        point_cloud_file = h5py.File(os.path.join(root, point_clouds_file), 'r')
        labels_file = h5py.File(os.path.join(root, labels_file), 'r')

        point_clouds = point_cloud_file['data'][:subset_size, :, :]
        identifiers = point_cloud_file['id'][:subset_size]
        joints = labels_file['real_world_coordinates'][:]

        point_cloud_file.close()
        labels_file.close()

        print("Time taken to load data from files: ", time.time() - start_time)

        start_time = time.time()

        # USE IDENTIFIES IN FORM XX_YYYYY (XX is subject number, YYYYY is frame number) to parse videos
        clip_index = 0
        max_frames = identifiers.shape[0]  # or set to a reasonable upper limit
        clip_points = collections.deque(maxlen=max_frames)
        clip_joints = collections.deque(maxlen=max_frames)

        for frame_idx in range(0, identifiers.shape[0] - 1):
            current_identifier = identifiers[frame_idx]
            next_identifier = identifiers[frame_idx + 1]

            clip_points.append(point_clouds[frame_idx])
            clip_joints.append(joints[frame_idx])

            # frames are not from the same sequence => finalize clip and add it to videos
            if current_identifier[:2] != next_identifier[:2]:
                n_frames = len(clip_points)
                self.videos.append(np.array(list(clip_points)))
                self.labels.append(np.array(list(clip_joints)))
                clip_points.clear()
                clip_joints.clear()

                for t in range(0, n_frames - frame_interval * (frames_per_clip - 1)):
                    self.index_map.append((clip_index, t))
                clip_index += 1

        print("Time taken to parse videos: ", time.time() - start_time)

        start_time = time.time()

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = frames_per_clip * np.prod(self.labels[0].shape[1:]) # 24 frames, 15 joints, 3D point dim

        print("Time taken to set class variables: ", time.time() - start_time)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t + i * self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        # here augmentations missing!

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        label = np.array([label[t + i * self.frame_interval] for i in range(self.frames_per_clip)])

        return clip.astype(np.float32), label.astype(np.float32), index


if __name__ == '__main__':
    dataset = ITOP(root='/data/iballester/datasets/ITOP/SIDE', frames_per_clip=16, train=False)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)