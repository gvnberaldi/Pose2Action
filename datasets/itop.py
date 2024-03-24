import os
import sys
import numpy as np
import h5py
import tqdm

from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'visualization'))

from plot_pc_joints import create_gif



class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, use_valid_only=False):
        super(ITOP, self).__init__()

        self.videos = []
        self.labels = []
        self.identifiers = []
        self.index_map = []

        # LOAD DATA FROM FILES
        point_clouds_folder = ''
        labels_file = ''

        if train:
            point_clouds_folder = os.path.join(root, "train")
            labels_file = "weakly_train_labels_37_14.h5"
        if not train:
            point_clouds_folder = os.path.join(root, "test")
            labels_file = "test_labels.h5"

        point_cloud_names = sorted(os.listdir(point_clouds_folder), key=lambda x: int(x.split('.')[0]))
        point_clouds = []

        type_name = "train" if train else "test"
        for pc_name in tqdm.tqdm(point_cloud_names, f"Loading {type_name} point clouds"):
            point_clouds.append(np.load(os.path.join(point_clouds_folder, pc_name))['arr_0'])

        labels_file = h5py.File(os.path.join(root, labels_file), 'r')

        identifiers = labels_file['id'][:]
        joints = labels_file['real_world_coordinates'][:]
        is_valid_flags = labels_file['is_valid'][:]

        labels_file.close()

        # USE IDENTIFIES IN FORM XX_YYYYY (XX is subject number, YYYYY is frame number) to parse videos
        clip_index = 0
        clip_points = []
        clip_joints = []
        clip_ids = []

        for frame_idx in range(0, identifiers.shape[0]):
            current_identifier = identifiers[frame_idx]
            next_identifier = -1

            if frame_idx != identifiers.shape[0] - 1:
                next_identifier = identifiers[frame_idx + 1]

            # add loaded point clouds if
            # a) use_valid_only flag set to false => use all available frames
            # b) use_valid_only flag set to true and valid bit with idx frame_idx is set to true
            # and the point cloud contains at least one point
            if (not use_valid_only or (use_valid_only and is_valid_flags[frame_idx] == 1)) and point_clouds[frame_idx].size > 0:
                clip_points.append(point_clouds[frame_idx])
                clip_joints.append(joints[frame_idx])
                clip_ids.append(current_identifier)

            # frames are not from the same sequence => finalize clip and add it to sequences
            # frame idx is the last idx => there's no next_identifies => still finalize the clip
            if frame_idx == identifiers.shape[0] - 1 or current_identifier[:2] != next_identifier[:2]:
                n_frames = len(clip_points)
                self.videos.append(clip_points)
                self.labels.append(clip_joints)
                self.identifiers.append(clip_ids)
                clip_points = []
                clip_joints = []
                clip_ids = []

                for t in range(0, n_frames - frame_interval * (frames_per_clip - 1)):
                    self.index_map.append((clip_index, t))
                clip_index += 1

        if use_valid_only:
            print(f"Using only frames labeled as valid. From the total of {len(point_clouds)} {type_name} frames using {len(self.index_map)} frames")

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
        identifiers = self.identifiers[index]

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
        ids = [identifiers[t + i * self.frame_interval] for i in range(self.frames_per_clip)]

        return clip.astype(np.float32), label.astype(np.float32), np.array([tuple(map(int, s.decode('utf-8').split('_'))) for s in ids])


if __name__ == '__main__':
    dataset = ITOP(root='/data/iballester/datasets/ITOP-CLEAN/SIDE', num_points=2048, frames_per_clip=1, train=True, use_valid_only=False)
    print(len(dataset))
    output_dir = 'visualization/gifs'
    clip, label, frame_idx = dataset[2662]
    print(clip.shape)
    #print(label)
    print(frame_idx)

    print(dataset.num_classes)

    create_gif(clip, label, frame_idx, output_dir)