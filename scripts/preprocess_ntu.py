# Python
import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
from multiprocessing import Pool

def process_video(video_path):
    video_name = os.path.basename(video_path)

    point_clouds = []
    for img_name in sorted(os.listdir(video_path)):
        img_path = os.path.join(video_path, img_name)
        img = imread(img_path)  # (H, W)

        depth_min = img[img > 0].min()
        depth_map = img

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]
        x = (x - W / 2) / focal * z
        y = (y - H / 2) / focal * z

        points = np.stack([x, y, z], axis=-1)
        point_clouds.append(points)

    np.savez_compressed(os.path.join(args.output, video_name + '.npz'), data=np.array(point_clouds, dtype=object))

parser = argparse.ArgumentParser(description='Depth to Point Cloud')
parser.add_argument('--input', default='/data/iballester/datasets/NTU60/nturgb+d_depth_masked', type=str)
parser.add_argument('--output', default='/data/iballester/datasets/ntu60/pc', type=str)
args = parser.parse_args()

W = 512
H = 424

xx, yy = np.meshgrid(np.arange(W), np.arange(H))
focal = 280

with Pool() as p:
    for n in range(1, 61):  # Loop over the range from 1 to 60
        print(f'Processing  A0{n}')
        video_paths = sorted(glob('%s/*A0%02d'%(args.input, n)))
        p.map(process_video, video_paths)