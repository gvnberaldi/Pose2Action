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

from augmentations.AugPipeline import AugPipeline
from visualization.plot_pc_joints import create_gif

class ITOP_p(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, use_valid_only=False,
                 aug_list=None, label_frame='middle'):
        super(ITOP_p, self).__init__()



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

        # Iterate over the joints_dict
        for id, (joint, is_valid) in joints_dict.items():
            # Check if the joint is valid and the identifier is greater than or equal to frames_per_clip
            if is_valid and int(id[-5:]) >= frames_per_clip:
                # Get the current frame and the previous frames_per_clip frames
                frames = [point_clouds_dict.get(id[:3] + str(int(id[-5:]) - i).zfill(5), None) for i in range(frames_per_clip)]
                # Add the joint and its corresponding frames to the valid_joints_dict
                self.valid_joints_dict[id] = (joint, frames)

        #Create a list of valid identifiers
        self.valid_identifiers = list(self.valid_joints_dict.keys())
        

        if use_valid_only:
            #self.valid_joints_count = sum(1 for joint, is_valid in joints_dict.values() if is_valid)
            print(f"Using only frames labeled as valid. From the total of {len(point_clouds)} {type_name} frames using {len(self.valid_identifiers)} valid joints")

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

        # Get the joint and frames from the valid_joints_dict
        joint, clip = self.valid_joints_dict.get(identifier, (None, [None] * self.frames_per_clip))

        # Check if the joint and frames are valid
        if joint is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joint or frames for identifier {identifier}")

        # Process the frames
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

    AUGMENT_TRAIN  = [

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
        },
        {
            "name": "CenterAug",
            "p_prob": 1.0,
            "p_axes": [True, True, True],
            "p_apply_extra_tensors": True
        },
    ]

    label_frame = 'last'

    dataset = ITOP(root='/data/iballester/datasets/ITOP-CLEAN/SIDE', num_points=4096, frames_per_clip=1, train=False, use_valid_only=True, aug_list=None ,label_frame=label_frame)
    dataset_p = ITOP_p(root='/data/iballester/datasets/ITOP-CLEAN/SIDE', num_points=4096, frames_per_clip=1, train=False, use_valid_only=True, aug_list=None ,label_frame=label_frame)

    
    

    clip, label, frame_idx = dataset[100]

    clip_p, label_p, frame_idx_p = dataset_p[100]

    output_dir = 'visualization/gifs'
    output_dir_p = 'visualization/gifs_p'


    create_gif(clip, label, frame_idx, output_dir, plot_lines=True, label_frame=label_frame)
    create_gif(clip_p, label_p, frame_idx_p , output_dir_p, plot_lines=True, label_frame=label_frame)
    
""" 
    # Ensure both datasets have the same length
    assert len(dataset) == len(dataset_p), "Datasets have different lengths"

    # Iterate over both datasets
    for i in range(len(dataset)):
        sample1 = dataset[i]
        sample2 = dataset_p[i]

        # Unpack the samples
        clip1, label1, frame_idx1 = sample1
        clip2, label2, frame_idx2 = sample2

        print(frame_idx1)
        print(frame_idx2)

        # Compare the clips
        if not np.array_equal(clip1, clip2):
            diff_indices = np.where(clip1 != clip2)
            print(clip1)
            print(clip2)
            break
            print(f"Clips at index {i} are not the same. They differ at the following indices: {diff_indices}")

        # Compare the labels
        if not np.array_equal(label1, label2):
            print(f"Labels at index {i} are not the same")


    print("All samples are the same") """