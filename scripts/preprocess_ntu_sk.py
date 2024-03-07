# Python
import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
from multiprocessing import Pool
from functools import partial
import cv2


def process_video(output_dir, video_path):

    W = 512
    H = 424

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    cx = 258
    cy = 207
    fx = 337
    fy = -337

    video_name = video_path.split('/')[-1]

    point_clouds = []
    for img_name in sorted(os.listdir(video_path)):
        img_path = os.path.join(video_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        depth_min = img[img > 0].min()
        depth_map = img

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]/1000
        x = (x - cx) / fx * z
        y = (y - cy) / fy * z

        points = np.stack([x, y, z], axis=-1)
        point_clouds.append(points)

    np.savez_compressed(os.path.join(output_dir, video_name + '.npz'), data=np.array(point_clouds, dtype=object))




def main():
    parser = argparse.ArgumentParser(description='Depth to Point Cloud')
    parser.add_argument('--input', default='/data/iballester/datasets/NTU60/nturgb+d_depth_masked', type=str)
    parser.add_argument('--output', default='/data/iballester/datasets/ntu60_sk/pc', type=str)
    args = parser.parse_args()


    with Pool() as p:      
        for n in range(1, 61):  # Loop over the range from 1 to 60
            print(f'Processing  A0{n}')
            video_paths = sorted(glob('%s/*A0%02d'%(args.input, n)))
            func = partial(process_video, args.output)
            p.map(func, video_paths)

if __name__ == '__main__':
    main()