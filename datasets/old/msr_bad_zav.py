import os
import sys
import numpy as np
#from utils import read_csv, get_video_split
from torch.utils.data import Dataset
from augmentations.AugPipeline import AugPipeline#
import augmentations_zav

DS_AUGMENTS_CFG = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 1,
            'p_min_angle' : 0.0,
            'p_max_angle' : 2.0*np.pi,
            
            'p_apply_extra_tensors': [False, False]
        },
        {
            'name': 'NoiseAug',
            'p_prob': 1.0,
            'p_stddev' : 0.005,
            'p_clip' : 0.02,
            'p_apply_extra_tensors': [False, False]
        },
        {
            'name': 'LinearAug',
            'p_prob':  1.0,
            'p_min_a' : 0.9,
            'p_max_a' : 1.1,
            'p_min_b' : 0.0,
            'p_max_b' : 0.0,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': [False, False]
        },
        {
            'name': 'MirrorAug',
            'p_prob': 1.0,
            'p_mirror_prob' : 0.5,
            'p_axes' : [True, False, True],
            'p_apply_extra_tensors': [False, False]
        }
    ]

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

            if label == 10 or label == 12:
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

                if label == 10 or label == 12:
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

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]
        
        if self.train:

            
            """ varying_frame_interval = np.random.randint(1, 10) + 1
           # Create a temporal t and update it to make sure it doesn't exceed the video length
            while True:
                varying_frame_interval=  varying_frame_interval - 1
                temp_t = t * varying_frame_interval
                if temp_t + (self.frames_per_clip * varying_frame_interval) < len(video):
                    break

            # Update t with the adjusted frame_interval
            t = temp_t """

            clip = [video[min(t+i*self.frame_interval, len(video)-1)] for i in range(self.frames_per_clip)] #Padding



            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
            clip = np.array(clip)

            """  # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

            angle = np.random.uniform(0, 360)
            clip = augmentations_zav.rotating_clip(clip, angle)
            clip = np.array(clip)

            prob_mirror_x = np.random.rand()
            if prob_mirror_x > 0.5:
                clip = augmentations_zav.mirror_x_axis_clip(clip)

            clip = np.array(clip)


            prob_mirror_y = np.random.rand()
            if prob_mirror_y > 0.5:
                clip = augmentations_zav.mirror_y_axis_clip(clip)

            clip = np.array(clip)

            clip = augmentations_zav.random_jitter(clip)

            clip = np.array(clip)  """


            aug_pipeline = AugPipeline()
            aug_pipeline.create_pipeline(self.aug)
            
            aug_clip, par, tensors = aug_pipeline.augment(clip)
                
            clip = np.array(aug_clip)   

        else:
            clip = [video[min(t+i*self.frame_interval, len(video)-1)] for i in range(self.frames_per_clip)] #Padding

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
    dataset = BAD(root='/data/iballester/datasets/BAD/f_depth_npz_0-8_7', frames_per_clip=24)
    clip, label, video_idx = dataset[0]
    #print(clip)
    #print(label)
    #print(video_idx)
    #print(dataset.num_classes)