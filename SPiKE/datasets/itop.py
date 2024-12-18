import os
import sys
import numpy as np
import h5py
import torch
import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from augmentations.AugPipeline import AugPipeline

class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, use_valid_only=False,
                 aug_list=None, label_frame='middle'):
        super(ITOP, self).__init__()

        self.label_frame = label_frame  
        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.root = root
        self.num_coord_joints = 45 # 15 joints, 3D point dim

        self._load_data(use_valid_only)

        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def _get_valid_joints(self, use_valid_only, joints_dict, point_clouds_dict):
        """Cumbersome but necessary logic to create clips of only valid joints and their corresponding frames."""

        valid_joints_dict = {}
        list_joints_items = list(joints_dict.items())

        # Process joints based on if we are considering only past frame(label_frame == 'last') or past and future frames (label_frame == 'middle')
        if self.label_frame == 'last':
            for id, (joint, is_valid) in joints_dict.items():

                # If we are only using valid joints, check if joint is valid and identifier is greater than or equal to frames_per_clip (so we have enough frames to add)
                if use_valid_only and is_valid and int(id[-5:]) >= self.frames_per_clip:
                    frames = [point_clouds_dict.get(id[:3] + str(int(id[-5:]) - self.frames_per_clip + 1 + i).zfill(5), None) for i in range(self.frames_per_clip)]
                    valid_joints_dict[id] = (joint, frames)
                # If not using valid_only, consider all joints
                elif not use_valid_only and int(id[-5:]) >= self.frames_per_clip:
                    frames = [point_clouds_dict.get(id[:3] + str(int(id[-5:]) - self.frames_per_clip + 1 + i).zfill(5), None) for i in range(self.frames_per_clip)]
                    valid_joints_dict[id] = (joint, frames)
        # If we are considering past and future frames, we need to ensure that we have enough frames before and after, and that they belong to the same person (see ITOP naming convention for more details)
        elif self.label_frame == 'middle':
            for i, (id, (joint, is_valid)) in enumerate(list_joints_items):
                # Get the identifier of the next frames_per_clip // 2 (i + frames_per_clip // 2 ) to make sure they belong to the same person
                next_half_frames_per_clip_id_person, next_item = (list_joints_items[i + self.frames_per_clip // 2] if i + self.frames_per_clip // 2 < len(list_joints_items) else (None, None))
                # If no next identifier available, we don't consider the current joint because we don't have enough frames
                if next_half_frames_per_clip_id_person is None:
                    continue
                # Check that the frames belong to the same person and that we have enough frames before and after
                if use_valid_only and is_valid and int(id[-5:]) >= self.frames_per_clip // 2 and int(id[:2]) == int(next_half_frames_per_clip_id_person[:2]):
                    middle_frame_starting_index = int(id[-5:]) - self.frames_per_clip // 2
                    frames = [point_clouds_dict.get(id[:3] + str(middle_frame_starting_index + i).zfill(5), None) for i in range(self.frames_per_clip)]
                    valid_joints_dict[id] = (joint, frames)
                # If not using valid_only, consider all joints (exceot for validity, same conditions as above)
                elif not use_valid_only and int(id[-5:]) >= self.frames_per_clip // 2 and int(id[:2]) == int(next_half_frames_per_clip_id_person[:2]):
                    middle_frame_starting_index = int(id[-5:]) - self.frames_per_clip // 2
                    frames = [point_clouds_dict.get(id[:3] + str(middle_frame_starting_index + i).zfill(5), None) for i in range(self.frames_per_clip)]
                    valid_joints_dict[id] = (joint, frames)
        return valid_joints_dict

    def _load_data(self, use_valid_only):

        point_clouds_folder = os.path.join(self.root, "train" if self.train else "test")
        labels_file = h5py.File(os.path.join(self.root, "train_labels.h5" if self.train else "test_labels.h5"), 'r')
        identifiers = labels_file['id'][:]
        joints = labels_file['real_world_coordinates'][:]
        is_valid_flags = labels_file['is_valid'][:]
        labels_file.close()

        point_cloud_names = sorted(os.listdir(point_clouds_folder), key=lambda x: int(x.split('.')[0]))
        point_clouds = []

        for pc_name in tqdm.tqdm(point_cloud_names, f"Loading {'train' if self.train else 'test'} point clouds"):
            point_clouds.append(np.load(os.path.join(point_clouds_folder, pc_name))['arr_0'])

        point_clouds_dict = {id.decode('utf-8'): point_clouds[i] for i, id in enumerate(identifiers)}
        joints_dict = {id.decode('utf-8'): (joints[i], is_valid_flags[i]) for i, id in enumerate(identifiers)}

        self.valid_joints_dict = self._get_valid_joints(use_valid_only, joints_dict, point_clouds_dict)
        self.valid_identifiers = list(self.valid_joints_dict.keys())

        if use_valid_only:
            print(f"Using only frames labeled as valid. From the total of {len(point_clouds)} {'train' if self.train else 'test'} frames using {len(self.valid_identifiers)} valid joints")

    def _process_frame(self, p):
        if p.shape[0] > self.num_points:
            r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
        elif p.shape[0] < self.num_points:
            repeat, residue = divmod(self.num_points, p.shape[0])
            r = np.concatenate([np.arange(p.shape[0])] * repeat + [np.random.choice(p.shape[0], size=residue, replace=False)], axis=0)
        else:
            return p
        return p[r, :]

    def __len__(self):
        return len(self.valid_identifiers)
    
    def __getitem__(self, idx):
        identifier = self.valid_identifiers[idx]
        joint, clip = self.valid_joints_dict.get(identifier, (None, [None] * self.frames_per_clip))

        if joint is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joint or frames for identifier {identifier}")

        clip = [self._process_frame(p) for p in clip]

        clip = torch.FloatTensor(clip)
        joint = torch.FloatTensor(joint).view(1, -1, 3)

        if self.aug_pipeline:
            clip, _, joint = self.aug_pipeline.augment(clip, joint)

        joint = joint.view(1, -1, 3)

        return clip, joint, np.array([tuple(map(int, identifier.split('_')))])