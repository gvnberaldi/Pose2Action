import os
import sys
import numpy as np
#from utils import read_csv, get_video_split
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

    
    def select_clip(self, video):

        #TODO: ADD RANDOMIZATION OF SAMPLING AND UPDATE THE FRAME INTERVAL
            #cambiar lo de los indices!
            #TODO no random sampling in testing

        clip_indices = []

        if len(video) < self.frames_per_clip:
            # Generate a random starting frame index between 0 and length of the video/self.frames_per_clip
            start_frame = random.randint(0, np.round(len(video) / self.frames_per_clip).astype(int))

            clip_ind = np.interp(np.linspace(start_frame, start_frame + self.frames_per_clip - 1, self.frames_per_clip), np.arange(len(video)), np.arange(len(video)))
            
            # Round the interpolated index to the nearest integer
            clip_ind = np.round(clip_ind).astype(int)
            
            # Create a clip by putting together the content of the video at the rounded index
            clip = [video[int(i)] for i in clip_ind]
        
        else:
            # Video is subsampled to adjust it to the number of frames per clip
            clip = [video[int(i)] for i in np.linspace(0, len(video) - 1, self.frames_per_clip)]

        return clip
    
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

            #clip = [video[min(t+i*self.frame_interval, len(video)-1)] for i in range(self.frames_per_clip)] #Padding
            clip = self.select_clip(video)


            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
            clip = np.array(clip)


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
        clip = clip / 30
        return clip.astype(np.float32), label, index, index

class BAD_ttt(Dataset):

    def __init__(self, root, frames_per_clip=16, frame_interval=1, max_frame_interval=20, num_points=2048, train=True, split_file='/data/iballester/datasets/BAD/sets_0-8_7.txt', aug=DS_AUGMENTS_CFG, num_slices=3):
        super(BAD_ttt, self).__init__()
        self.max_frame_interval = max_frame_interval
        self.videos, self.labels, self.index_map = [], [], []
        index = 0
        self.split_info = read_csv(split_file)
        self.aug = aug
        self.num_slices = num_slices

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
    
    def select_clip(self, video):

        #TODO: ADD RANDOMIZATION OF SAMPLING AND UPDATE THE FRAME INTERVAL
            #cambiar lo de los indices!
            #TODO no random sampling in testing

        clip_indices = []

        if len(video) < self.frames_per_clip:
            # Generate a random starting frame index between 0 and length of the video/self.frames_per_clip
            start_frame = random.randint(0, np.round(len(video) / self.frames_per_clip).astype(int))

            clip_ind = np.interp(np.linspace(start_frame, start_frame + self.frames_per_clip - 1, self.frames_per_clip), np.arange(len(video)), np.arange(len(video)))
            
            # Round the interpolated index to the nearest integer
            clip_ind = np.round(clip_ind).astype(int)
            
            # Create a clip by putting together the content of the video at the rounded index
            clip = [video[int(i)] for i in clip_ind]
        
        else:
            # Video is subsampled to adjust it to the number of frames per clip
            clip = [video[int(i)] for i in np.linspace(0, len(video) - 1, self.frames_per_clip)]

        return clip

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = self.select_clip(video)
        
        if self.train:

            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
            clip = np.array(clip)

            aug_pipeline = AugPipeline()
            aug_pipeline.create_pipeline(self.aug)
            
            aug_clip, par, tensors = aug_pipeline.augment(clip)
                
            clip = np.array(aug_clip)  
            
        else:

            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                clip[i] = p[r, :]
            clip = np.array(clip)

        clip = clip / 30

        sliced_frames = np.array_split(clip, self.num_slices)

        # Generate an index mapping to keep track of the shuffling
        index_mapping = np.random.permutation(self.num_slices)

        # Shuffle the sets based on the shuffled index mapping
        shuffled_frames = [sliced_frames[i] for i in index_mapping]

        # Concatenate the shuffled sets
        shuffled_clip = np.concatenate(shuffled_frames, axis=0)

        # Convert to NumPy arrays
        shuffled_clip = np.array(shuffled_clip)
        
        return shuffled_clip.astype(np.float32), label, index, index_mapping


class BAD12(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, max_frame_interval=20, num_points=2048, train=True, split_file='/data/iballester/datasets/BAD/sets_0-8_7.txt', aug=DS_AUGMENTS_CFG):
        super(BAD12, self).__init__()
        self.max_frame_interval = max_frame_interval
        self.videos, self.labels, self.index_map = [], [], []
        index = 0
        self.aug = aug

        for root, split_file in zip(root, split_file):
            self._process_directory(root, split_file, frames_per_clip, frame_interval, num_points, train, index)
            index += 1


                
        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1
    
    def _process_directory(self, root, split_file, frames_per_clip, frame_interval, num_points, train, index):
        self.split_info = read_csv(split_file)

        for video_name in os.listdir(root):
            if video_name.startswith("."):
                continue

            split = get_video_split(video_name, self.split_info)
            label = int(video_name.split('_')[3])

            if (split == "train") and train:
                if label == 10 or label ==13 or label == 12:
                    continue
                #print(label)
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']

                self.videos.append(video)
                self.labels.append(label)

                nframes = video.shape[0]

                # Loop for the regular frames
                for t in range(0, nframes, frame_interval):
                    self.index_map.append((index, t))
                index += 1

            if (split == "val" or split == "test") and not train:
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
        clip = clip / 30
        return clip.astype(np.float32), label, index, index

if __name__ == '__main__':
    dataset = BAD(root='/data/iballester/datasets/BAD/f_depth_npz_0-8_7', frames_per_clip=24)
    clip, label, video_idx = dataset[0]
    #print(clip)
    #print(label)
    #print(video_idx)
    #print(dataset.num_classes)