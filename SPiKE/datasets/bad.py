import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from SPiKE.SPiKE.augmentations.AugPipeline import AugPipeline


class BAD(Dataset):
    def __init__(self, root=None, num_points=2048, train=True, frames_per_clip=16, frame_interval=1, aug_list=None, labeled_frame='last'):
        super(BAD, self).__init__()
        self.root = root
        if root is None:
            raise ValueError(f"Invalid root path: {root}")
        self.num_points = num_points
        self.train = train
        self.num_coord_joints = 45  # 15 joints, 3D point dim
        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.labeled_frame = labeled_frame
        self.identifiers = None
        self.valid_identifiers = None
        self.data = None
        self.joint_frames_dict = None
        self._load_data()

        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def _create_joint_frames_dict(self):
        joint_frames_dict = {}

        for id, (point_cloud, joint) in self.data.items():
            subject_id = id.split('_')[0]
            frame_number = int(id.split('_')[1])
            clip = None
            if self.labeled_frame == 'last':
                clip = [
                    self.data.get(f'{subject_id}_{str(frame_number - self.frames_per_clip + i + 1).zfill(5)}',
                                  (None, None))[0]
                    for i in range(0, self.frames_per_clip, self.frame_interval)
                ]
            elif self.labeled_frame == 'middle':
                half_clip = self.frames_per_clip // 2
                clip = [
                    self.data.get(f'{subject_id}_{str(frame_number + i).zfill(5)}',
                                  (None, None))[0]
                    for i in range(-half_clip, half_clip + 1, self.frame_interval)
                ]

            if clip and all(frame is not None for frame in clip):
                joint_frames_dict[id] = (joint, clip)

        return joint_frames_dict

    def _load_data(self):
        point_clouds_file = h5py.File(os.path.join(self.root, "train_point_clouds.h5" if self.train else "test_point_clouds.h5"), 'r')
        labels_file = h5py.File(os.path.join(self.root, "train_labels.h5" if self.train else "test_labels.h5"), 'r')

        point_clouds = point_clouds_file['data'][:]
        identifiers = labels_file['id'][:]
        joints = labels_file['3d_joints_coordinates'][:]
        point_clouds_file.close()
        labels_file.close()

        # Create a dictionary with key equal to the identifier
        # and value the tuple of the corresponding point cloud and joint
        data_dict = {ident.decode('utf-8'): (point_cloud, joint) for ident, point_cloud, joint in zip(identifiers, point_clouds, joints)}

        self.identifiers = identifiers
        self.data = data_dict
        self.joint_frames_dict = self._create_joint_frames_dict()
        self.valid_identifiers = list(self.joint_frames_dict.keys())

    def _normalize_ids(self, ids):
        # Split each ID to extract the prefix and the numeric part
        decoded_ids = [id.decode('utf-8') for id in ids]
        parts = [id.split('_') for id in decoded_ids]

        # Generate new IDs starting from 1, keeping the original prefix
        normalized_ids = []
        for i, (prefix, num) in enumerate(parts, start=1):
            new_id = f"{prefix}_{i:05d}"
            normalized_ids.append(new_id)

        return normalized_ids

    def _reshape_point_cloud(self, p):
        if p.shape[0] > self.num_points:
            r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
        elif p.shape[0] < self.num_points:
            repeat, residue = divmod(self.num_points, p.shape[0])
            r = np.concatenate([np.arange(p.shape[0])] * repeat + [np.random.choice(p.shape[0], size=residue, replace=False)], axis=0)
        else:
            return p
        return p[r, :]

    def __getitem__(self, idx):
        identifier = self.valid_identifiers[idx]
        joint, clip = self.joint_frames_dict.get(identifier, (None, [None] * self.frames_per_clip))

        if joint is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joint or frames for identifier {identifier}")

        clip = [self._reshape_point_cloud(p) for p in clip]

        clip = torch.FloatTensor(np.array(clip))
        joint = torch.FloatTensor(joint).view(1, -1, 3)

        if self.aug_pipeline:
            clip, _, joint = self.aug_pipeline.augment(clip, joint)

        joint = joint.view(1, -1, 3)

        return clip, joint, np.array([tuple(map(int, identifier.split('_')))])

    def __len__(self):
        return len(self.valid_identifiers)
