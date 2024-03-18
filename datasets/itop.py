import os
import sys
import numpy as np
import h5py
import tqdm

from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../visualization'))

from plot_pc_joints import create_gif


def farthest_point_sampling(points, num_samples):
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = cdist(farthest_pts[0:1], points).squeeze()
    for i in range(1, num_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, cdist(farthest_pts[i:i+1], points).squeeze())
    return farthest_pts

class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True):
        super(ITOP, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []

        # LOAD DATA FROM FILES
        point_clouds_folder = ''
        labels_file = ''

        if train:
            point_clouds_folder = os.path.join(root, "train")
            labels_file = "train_labels.h5"
        if not train:
            point_clouds_folder = os.path.join(root, "test")
            labels_file = "test_labels.h5"

        point_cloud_names = sorted(os.listdir(point_clouds_folder), key=lambda x: int(x.split('.')[0]))
        point_clouds = []

        t = "train" if train else "test"

        labels_file = h5py.File(os.path.join(root, labels_file), 'r')
        for pc_name in tqdm.tqdm(point_cloud_names, f"Loading {t} point clouds"):
            point_clouds.append(np.load(os.path.join(point_clouds_folder, pc_name))['arr_0'])

        identifiers = labels_file['id'][:]
        joints = labels_file['real_world_coordinates'][:]
        is_valid = labels_file['is_valid'][:]

        labels_file.close()

        # USE IDENTIFIES IN FORM XX_YYYYY (XX is subject number, YYYYY is frame number) to parse videos
        clip_index = 0
        clip_points = []
        clip_joints = []
        for frame_idx in range(0, identifiers.shape[0] - 1):
            if is_valid[frame_idx]:
                current_identifier = identifiers[frame_idx]
                next_identifier = identifiers[frame_idx + 1]

                clip_points.append(point_clouds[frame_idx])
                clip_joints.append(joints[frame_idx])

                # frames are not from the same sequence => finalize clip and add it to sequences
                if current_identifier[:2] != next_identifier[:2]:
                    n_frames = len(clip_points)
                    self.videos.append(clip_points)
                    self.labels.append(clip_joints)
                    clip_points = []
                    clip_joints = []

                    for t in range(0, n_frames - frame_interval * (frames_per_clip - 1)):
                        self.index_map.append((clip_index, t))
                    clip_index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = frames_per_clip * np.prod(self.labels[0][0].shape) # X frames, 15 joints, 3D point dim

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t + i * self.frame_interval] for i in range(self.frames_per_clip)]

        #FPS
        """for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                clip[i] = farthest_point_sampling(p, self.num_points)
            elif p.shape[0] < self.num_points:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
        clip = np.array(clip)"""

        # Random
        clip = [video[t + i * self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            elif p.shape[0] == self.num_points: # TODO: very ugly, try to make nicer
                clip[i] = p
                continue
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        
        # here augmentations missing!
        """
        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300
        """

        label = np.array([label[t + i * self.frame_interval] for i in range(self.frames_per_clip)])

        return clip.astype(np.float32), label.astype(np.float32), index


if __name__ == '__main__':
    dataset = ITOP(root='/data/iballester/datasets/ITOP_old/ITOP-PREP-COMPL/SIDE', num_points=4096, frames_per_clip=20, train=False)
    #print(len(dataset))
    output_dir = 'visualization/gifs'
    clip, label, video_idx = dataset[300]
    #print(clip)
    #print(label)
    #print(video_idx)
    print(dataset.num_classes)

    create_gif(clip, label, video_idx, output_dir)
