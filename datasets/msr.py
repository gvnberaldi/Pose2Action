import os
import sys
import numpy as np
from torch.utils.data import Dataset
import random

class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=16, frame_interval=1, num_points=2048, train=True):
        super(MSRAction3D, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
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


        if len(video) < self.frames_per_clip:
            # Generate a random starting frame index between 0 and length of the video/self.frames_per_clip
            start_frame = random.randint(0, len(video) / self.frames_per_clip)

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

        #clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300

        return clip.astype(np.float32), label, index, []

if __name__ == '__main__':
    dataset = MSRAction3D(root='/data/iballester/datasets/MSRAction3D_Output/', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
