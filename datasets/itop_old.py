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

class ITOP(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True, use_valid_only=False,
                 aug_list=None, label_frame='middle'):
        super(ITOP, self).__init__()


        self.videos = []
        self.labels = []
        self.identifiers = []
        self.index_map = []
        self.label_frame = label_frame
        index = 0
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

        type_name = "train" if train else "test"
        for pc_name in tqdm.tqdm(point_cloud_names, f"Loading {type_name} point clouds"):
            point_clouds.append(np.load(os.path.join(point_clouds_folder, pc_name))['arr_0'])

        labels_file = h5py.File(os.path.join(root, labels_file), 'r')

        identifiers = labels_file['id'][:]
        joints = labels_file['real_world_coordinates'][:]
        is_valid_flags = labels_file['is_valid'][:]

        labels_file.close()

        # First loop: Add only valid frames to a list
        valid_frames = []
        for frame_idx in range(0, identifiers.shape[0]):
            if (not use_valid_only or (use_valid_only and is_valid_flags[frame_idx] == 1)) and point_clouds[frame_idx].size > 0:
            #if train or (not train and is_valid_flags[frame_idx] == 1) and point_clouds[frame_idx].size > 0:
                valid_frames.append((point_clouds[frame_idx], joints[frame_idx], identifiers[frame_idx]))

        # Second loop: Create videos from contiguous frames
        video_points = []
        video_joints = []
        video_ids = []
        for frame_idx in range(0, len(valid_frames)):
            video_points.append(valid_frames[frame_idx][0])
            video_joints.append(valid_frames[frame_idx][1])
            video_ids.append(valid_frames[frame_idx][2])

            if frame_idx == len(valid_frames) - 1 or int(valid_frames[frame_idx][2][-5:]) + 1 != int(valid_frames[frame_idx + 1][2][-5:]):
                n_frames = len(video_points)
                self.videos.append(video_points)
                self.labels.append(video_joints)
                self.identifiers.append(video_ids)
                video_points = []
                video_joints = []
                video_ids = []

        if use_valid_only:
            print(f"Using only frames labeled as valid. From the total of {len(point_clouds)} {type_name} frames using {len(valid_frames)} frames")

        #Third loop: Create index map
        for video in self.videos:
            n_frames = len(video)
            for t in range(n_frames):
                self.index_map.append((index, t))
            index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = np.prod(self.labels[0][0].shape) # 15 joints, 3D point dim
        
        # Create augmentation pipeline
        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]
        identifiers = self.identifiers[index]

        clip = [] 
        clip_label = []  
        clip_ids = []  

        # If the video length is shorter than frames_per_clip, append the frames that exist and then pad with the last frame
        if len(video) < self.frames_per_clip:
            clip.extend(video[i] if i < len(video) else video[-1] for i in range(self.frames_per_clip))
            clip_label.extend(label[i] if i < len(video) else label[-1] for i in range(self.frames_per_clip))
            clip_ids.extend(identifiers[i] if i < len(video) else identifiers[-1] for i in range(self.frames_per_clip))
        # If the video length is equal to or longer than frames_per_clip, append the last possible full clip
        else:
            last_possible_clip_start = len(video) - self.frames_per_clip*self.frame_interval
            clip.extend(video[min(t+i*self.frame_interval, last_possible_clip_start + i*self.frame_interval)] for i in range(self.frames_per_clip))
            clip_label.extend(label[min(t+i*self.frame_interval, last_possible_clip_start + i*self.frame_interval)] for i in range(self.frames_per_clip))
            clip_ids.extend(identifiers[min(t+i*self.frame_interval, last_possible_clip_start + i*self.frame_interval)] for i in range(self.frames_per_clip))
            

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
        #clip = np.array(clip)
            
        clip = torch.FloatTensor(clip)
        clip_label = torch.FloatTensor(clip_label)

        #print("clip_label.shape: ", clip_label.shape)

        if self.label_frame == 'middle': 
            middle_frame_index = clip_label.shape[0] // 2  # Find the index of the middle frame
            clip_label = clip_label[np.newaxis,middle_frame_index,:]  # Select only the middle frame from the target tensor
        elif self.label_frame == 'last':
            clip_label = clip_label[np.newaxis,-1,:]

            

        #print("clip_label.shape2: ", clip_label.shape)



        if self.aug_pipeline is not None:
            clip, _, clip_label = self.aug_pipeline.augment(clip, clip_label)

        #Extend clip_label to be (1,15,3)
        clip_label = clip_label.view(1, -1, 3)

        return clip, clip_label, np.array([tuple(map(int, s.decode('utf-8').split('_'))) for s in clip_ids])


if __name__ == '__main__':

    AUGMENT_TEST  = [
        {
            "name": "CenterAug",
            "p_prob": 1.0,
            "p_axes": [True, True, True],
            "p_apply_extra_tensors": True
        }]


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

    label_frame = 'middle'

    dataset = ITOP(root='/data/iballester/datasets/ITOP-CLEAN/SIDE', num_points=4096, frames_per_clip=1, train=False, use_valid_only=False, aug_list=AUGMENT_TEST ,label_frame=label_frame)
    print('len dataset: ', len(dataset))
    print(len(dataset.videos))
    print(len(dataset.labels))
    print(len(dataset.identifiers))
    print(len(dataset.index_map))

    output_dir = 'visualization/gifs'
    clip, label, frame_idx = dataset[3013]
    print(clip.shape)
    
    #print(label)
    print(frame_idx)

    print(dataset.num_classes)

    create_gif(clip, label, frame_idx, output_dir, plot_lines=True, label_frame=label_frame)