import os
import sys
import numpy as np
import h5py
import torch
import tqdm

from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'visualization'))
sys.path.append(os.path.join(ROOT_DIR, 'augmentations'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from augmentations.AugPipeline import AugPipeline
from visualization.plot_pc_joints import create_gif, clean_create_gif

class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, use_valid_only=False,
                 aug_list=None, label_frame='middle'):
        super(ITOP, self).__init__()



        self.label_frame = label_frame
        # LOAD DATA FROM FILES
        point_clouds_folder = ''
        labels_file = ''

        if train:
            point_clouds_folder = os.path.join(root, "train")
            labels_file = "train_labels.h5"
        if not train:
            point_clouds_folder = os.path.join(root, "test")
            labels_file = "test_labels.h5"

        labels_file = h5py.File(os.path.join(root, labels_file), 'r')

        identifiers = labels_file['id'][:]
        joints = labels_file['real_world_coordinates'][:]
        is_valid_flags = labels_file['is_valid'][:]

        labels_file.close()

        point_cloud_names = sorted(os.listdir(point_clouds_folder), key=lambda x: int(x.split('.')[0]))
        point_clouds = []


        type_name = "train" if train else "test"
        for pc_name in tqdm.tqdm(point_cloud_names, f"Loading {type_name} point clouds"):
            point_clouds.append(np.load(os.path.join(point_clouds_folder, pc_name))['arr_0'])

        point_clouds_dict = {id.decode('utf-8'): point_clouds[i] for i, id in enumerate(identifiers)}
        joints_dict = {id.decode('utf-8'): (joints[i], is_valid_flags[i]) for i, id in enumerate(identifiers)}

        self.valid_joints_dict = {}

        if self.label_frame=='last':

            # Iterate over the joints_dict
            for id, (joint, is_valid) in joints_dict.items():
                # Check if the joint is valid and the identifier is greater than or equal to frames_per_clip
                if use_valid_only:
                    if is_valid and int(id[-5:]) >= frames_per_clip-1:
                        # Get the current frame and the previous frames_per_clip frames
                        frames = [point_clouds_dict.get(id[:3] + str(int(id[-5:]) - frames_per_clip + 1 + i).zfill(5), None) for i in range(frames_per_clip)]
                        # Add the joint and its corresponding frames to the valid_joints_dict
                        self.valid_joints_dict[id] = (joint, frames)
                else: 
                    if int(id[-5:]) >= frames_per_clip-1:
                        frames = [point_clouds_dict.get(id[:3] + str(int(id[-5:]) - frames_per_clip + 1 + i).zfill(5), None) for i in range(frames_per_clip)]
                        self.valid_joints_dict[id] = (joint, frames)


            #Create a list of valid identifiers
            self.valid_identifiers = sorted(self.valid_joints_dict.keys())
            self.valid_identifiers = list(self.valid_joints_dict.keys())

            if use_valid_only:
                #self.valid_joints_count = sum(1 for joint, is_valid in joints_dict.values() if is_valid)
                print(f"Using only frames labeled as valid. From the total of {len(point_clouds)} {type_name} frames using {len(self.valid_identifiers)} valid joints")
        
        else:
            raise ValueError(f"Middle frame not implemented yet")  

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = 45 # 15 joints, 3D point dim
        
        # Create augmentation pipeline
        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def __len__(self):
        return len(self.valid_identifiers)
    
    def __getitem__(self, idx):


        identifier = self.valid_identifiers[idx]
        joint, clip = self.valid_joints_dict.get(identifier, (None, [None] * self.frames_per_clip))

        if joint is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joint or frames for identifier {identifier}")

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

        
        # Convert clip and joint to tensors
        clip = torch.FloatTensor(clip)
        joint = torch.FloatTensor(joint)

        # Extend joint to be (1,15,3)
        joint = joint.view(1, -1, 3)

        if self.aug_pipeline is not None:
            clip, _, joint = self.aug_pipeline.augment(clip, joint)

        # Extend joint to be (1,15,3)
        joint = joint.view(1, -1, 3)

        return clip, joint, np.array([tuple(map(int, identifier.split('_')))])

if __name__ == '__main__':

    AUGMENT_TEST  = [
    {
        "name": "CenterAug",
        "p_prob": 1.0,
        "p_axes": [True, True, True],
        "apply_on_gt": True
    }]

    AUGMENT_TRAIN  = [

        {
            "name": "CenterAug",
            "p_prob": 1.0,
            "p_axes": [True, True, True],
            "p_apply_extra_tensors": True
        },

        {
            "name": "RotationAug",
            "p_prob": 1.0,
            "p_axis": 1,
            "p_min_angle": 1.57,
            "p_max_angle": 1.57,
            "p_apply_extra_tensors": True
        },
        {
            "name": "NoiseAug",
            "p_prob": 0.0,
            "p_stddev": 0.01,
            "p_clip": 0.02,
            "p_apply_extra_tensors": False
        },
        {
            "name": "LinearAug",
            "p_prob": 0.0,
            "p_min_a": 0.9,
            "p_max_a": 1.1,
            "p_min_b": 0.0,
            "p_max_b": 0.0,
            "p_channel_independent": True,
            "p_apply_extra_tensors": True
        },
        {
            "name": "MirrorAug",
            "p_prob": 0.0,
            "p_axes": [True, False, False],
            "p_apply_extra_tensors": True
        },
        {
            "name": "MirrorAug",
            "p_prob": 0.0,
            "p_axes": [False, False, True],
            "p_apply_extra_tensors": True
        }
    ]

    label_frame = 'last'

    dataset_p = ITOP(root='/data/iballester/datasets/ITOP-CLEAN/SIDE', num_points=4096, frames_per_clip=5, train=False, use_valid_only=False, aug_list=AUGMENT_TEST, label_frame=label_frame)

    clip, label, frame_idx = dataset_p[3003]

    output_dir = 'visualization/gifs'

    create_gif(clip, label, frame_idx, output_dir, plot_lines=True, label_frame=label_frame)
    
