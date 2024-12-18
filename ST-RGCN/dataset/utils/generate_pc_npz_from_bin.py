import struct
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/Volumes/Irene_CVL2/BathroomActivitiesDataset_old/BAD2/f_depth_bin', help='Depth directory for input [default: Depth]')
parser.add_argument('--output_dir', default='/Volumes/Irene_CVL2/BAD-dataset/BAD2/pc', help='Output processed data directory [default: processed_data]')
parser.add_argument('--num_cpu', type=int, default=6, help='Number of CPUs to use in parallel [default: 6]')
FLAGS = parser.parse_args()

input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
num_cpu = FLAGS.num_cpu

extrinsic = [[-7.16662232e-01, -6.97410078e-01,  3.79845046e-03,  7.06164513e+00],
 [-5.66819662e-01,  5.85622814e-01,  5.79449213e-01, -6.77704956e+00],
 [-4.06338180e-01,  4.13116330e-01, -8.14999498e-01,  1.66869640e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

focal = 554

#intrinsic = np.array([[554, 0.0, 320],
#                      [0.0, -554, 240],
#                      [0.0, 0.0, 1.0]])

def read_bin(filename: Path) -> np.ndarray:
    with filename.open('rb') as f:
        num_frames, width, height = struct.unpack("<LLL", f.read(12))
        depth_data = f.read(num_frames * height * width * 4)
    depth = np.frombuffer(depth_data, dtype=np.uint32)
    depth = np.reshape(depth, (num_frames, height, width))
    return depth

def set_pointcloud(pointcloud,extrinsic):

    inv_extrinsic = np.linalg.inv(extrinsic)
    data_rotation = inv_extrinsic[:3, :3]
    data_translation = inv_extrinsic[:3, [3]]
    data = (data_rotation @ pointcloud) + data_translation
    data = data / 100  # Scale the point cloud by 100 
    return data

def process_one_file(filename: Path):
    output_filename = filename.stem + '.npz'
    output_path = output_dir / output_filename

    if output_path.exists():
        print(f"Output file already exists for {filename}. Skipping processing.")
        return

    depth = read_bin(filename)
    xx, yy = np.meshgrid(np.arange(depth.shape[2]), np.arange(depth.shape[1]))
    point_clouds = []

    for d in range(depth.shape[0]):
        depth_map = depth[d]
        if len(depth_map[depth_map > 0]) > 0:
            x = xx[depth_map > 0]
            y = yy[depth_map > 0]
            z = depth_map[depth_map > 0]
            x = (x - depth_map.shape[1] / 2) / focal * z
            y = (y - depth_map.shape[0] / 2) / (-focal) * z
            points = np.stack([x, y, z], axis=-1)
            pointcloud_set = set_pointcloud(points.T, extrinsic=extrinsic)
            pointcloud_set = np.array(pointcloud_set, dtype=np.float16)
            point_clouds.append(pointcloud_set.T)

    point_clouds = np.array(point_clouds, dtype=object)
    np.savez_compressed(output_path, point_clouds=point_clouds)

output_dir = Path(output_dir)
input_dir = Path(input_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = list(input_dir.glob('*'))
total_files = len(files)

for i, input_file in enumerate(files):
    if input_file.name == ".DS_Store":
        continue
    print(f"Processing file {i+1} of {total_files}: {input_file.name}")
    process_one_file(input_file)