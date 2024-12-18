import os
import glob
from itertools import product

import imageio
import numpy as np
from scipy.linalg import sqrtm
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist


extrinsic = [[ 9.93092396e-01, -1.16127039e-01, -1.67929840e-02, -1.08303368e+01],
                [-9.37719402e-02, -8.71518646e-01,  4.81312864e-01,  4.60539354e+00],
                [-7.05288364e-02, -4.76413434e-01, -8.76387998e-01,  2.57003849e+01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]


sensor_intrinsic = np.array([
    [554, 0.0, 320],
    [0.0, -554, 240],
    [0.0, 0.0, 1.0]
])


def compute_pointcloud(frame):
    img_shape = frame.shape
    xy_img_points = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    points = np.concatenate([*xy_img_points, frame]).reshape(3, -1)

    points[[0, 1]] *= points[[2]]
    points = np.linalg.inv(sensor_intrinsic) @ points
    return points.reshape(3, *img_shape)

def get_triangulation(pointcloud, threshold=10):
    pointcloud = pointcloud.transpose((1, 2, 0))
    columns = pointcloud.shape[1]
    ij_index = lambda i, j: i*columns+j
    triangulation = []
    for i in range(pointcloud.shape[0]-1):
        for j in range(pointcloud.shape[1]-1):
            a, b, c, d = pointcloud[i, j], pointcloud[i, j+1], pointcloud[i+1, j], pointcloud[i+1, j+1]
            if np.max(pdist(np.array([a, b, d]))) < threshold:
                triangulation.append([ij_index(i, j), ij_index(i+1, j+1), ij_index(i, j+1)])
            if np.max(pdist(np.array([a, c, d]))) < threshold:
                triangulation.append([ij_index(i, j), ij_index(i+1, j), ij_index(i+1, j+1)])

    triangulation = np.array(triangulation)
    return triangulation

def get_average_frame(directory, n=100, median=True):
    filename_list = glob.glob(os.path.join(directory, '*.png'))
    filename_list.sort()

    frame_stack = np.array([imageio.imread(filename) for filename in filename_list[-n:]])
    if median:
        median_frame = np.median(frame_stack, axis=0).astype(np.uint16)
        return median_frame
    else:
        shape = frame_stack.shape[1:]
        mean_frame = np.zeros(shape)
        for i, j in product(np.arange(shape[0]), np.arange(shape[1])):
            column = frame_stack[:, i, j]
            lb, ub = np.percentile(column, [20, 80])
            mask = np.logical_and(column >= lb, column <= ub)
            mean_frame[i, j] = np.mean(column[mask])
        mean_frame = mean_frame.astype(np.uint16)
        return mean_frame

def orthogonalize_matrix(m):
    # nearest orthogonal matrix
    m_ortho = m @ np.linalg.inv(sqrtm(m.T @ m)).real
    # print(m_ortho @ m_ortho.T) # test if it works (it does)
    return m_ortho

class DepthViewer:
    def __init__(self, frame, layout_size=None, layout_offset=(0, 0), initial_extrinsic=None, stride=1,
            cmap='rainbow', show_marker=True, show_layout=True):
        self.frame = frame
        self.cmap = cmap
        self.show_marker = show_marker
        self.show_layout = show_layout
        self.pointcloud, self.pointcloud_colors = self.get_pointcloud(frame, stride)

        if layout_size is None:
            layout_shape = (3, int(self.pointcloud[0].ptp()), int(self.pointcloud[1].ptp()))
            layout = np.ones(layout_shape) * 0.8 # light gray
        else:
            layout = np.ones((3, *layout_size)) * 0.8

        triangulation = get_triangulation(self.pointcloud.reshape((3, *frame.shape)), threshold=20)
        self.triangulation = o3d.utility.Vector3iVector(triangulation)
        self.n_draw_modes = 3

        x_mesh, y_mesh = np.meshgrid(np.arange(layout.shape[2]) - layout_offset[0], np.arange(layout.shape[1]) - layout_offset[1])
        layout_points = np.stack((x_mesh, y_mesh, np.zeros(x_mesh.shape))).reshape((3, -1))
        layout_colors = o3d.utility.Vector3dVector(layout.reshape(3, -1).T)
        self.layout = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(layout_points.T))
        self.layout.colors = layout_colors

        self.needs_redraw = True

        self.adjust_accuracy = 1
        self.parameter_mode = 0 # 0..6
        self.draw_mode = 0 # 0 is pointcloud, 1 mesh, 2 smoothed mesh
        self.marker_mode = False
        self.marker_position = np.array([0., 0., 0.])

        if initial_extrinsic is None:
            initial_extrinsic = np.eye(4)

        inv_extrinsic = np.linalg.inv(initial_extrinsic)
        self.data_rotation = inv_extrinsic[:3, :3]
        self.data_translation = inv_extrinsic[:3, [3]]

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Depth Viewer', width=1000, height=1000, left=100, top=100)

        mode_keys = [88, 89, 90, 80, 74, 82, 65] # X, Y, Z, P(pitch), J(yaw), R(roll), adjust accuracy (A)
        mode_names = ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'accuracy']
        adjust_keys = [45, 61] # minus, equals(plus)
        other_keys = [83, 84, 77] # (S)how matrix and translation, (T)oggle draw mode, (M)arker mode toggle
        print('''
        The following parameters are adjustable: 
            X-axis (z), Y-axis (y), Z-axis (z), Pitch (p), Yaw (j), Roll (r), Adjustment accuracy (a)
        
        Parameters are adjustable using the + and the - key.
        
        Further hotkeys are:
            Show current extrinsic matrix, translation and rotation (S)
            Toggle the draw mode between pointcloud and triangulation (T)
            Turn on/off marker mode (M)
        ''')

        def callback(key_code, _):
            self.needs_redraw = True

            if key_code in mode_keys:
                self.parameter_mode = mode_keys.index(key_code)
                print('adjust mode is', mode_names[self.parameter_mode])
                return
            if key_code in other_keys:
                if key_code == 83:
                    print('rotation:', self.data_rotation)
                    print('translation:', self.data_translation[:, 0])
                    print('extrinsic:', np.linalg.inv(np.concatenate([
                        np.concatenate([self.data_rotation, self.data_translation], axis=1),
                        np.array([[0., 0., 0., 1.]])
                    ], axis=0)))
                    print('marker position:', self.marker_position)

                    data = (self.data_rotation @ self.pointcloud) + self.data_translation
                    dists = np.linalg.norm(data - self.marker_position.reshape((-1, 1)), axis=0)
                    pixel_index = np.unravel_index(np.argmin(dists), self.frame.shape)
                    print('closest index to marker:', pixel_index)
                if key_code == 84:
                    self.draw_mode = (self.draw_mode + 1) % self.n_draw_modes
                if key_code == 77:
                    self.marker_mode = not self.marker_mode
                    print('marker mode:', self.marker_mode)
                return

            if key_code == 45:
                change = -self.adjust_accuracy
            else: # key_code = 61
                change = self.adjust_accuracy

            if not self.marker_mode:
                if self.parameter_mode in [0, 1, 2]:
                    self.data_translation[self.parameter_mode] += change
                elif self.parameter_mode in [3, 4, 5]:
                    rotation_mode = ['x', 'z', 'y'][self.parameter_mode-3]
                    self.data_rotation = Rotation.from_euler(rotation_mode, change, degrees=True).as_matrix() @ self.data_rotation
                    self.data_rotation = orthogonalize_matrix(self.data_rotation)
            else:
                if self.parameter_mode in [0, 1, 2]:
                    self.marker_position[self.parameter_mode] += change

            if self.parameter_mode == 6:
                if change > 0:
                    self.adjust_accuracy *= 2
                else:
                    self.adjust_accuracy /= 2
                print('accuracy:', self.adjust_accuracy)

        def make_callback(key_code):
            return lambda *args: callback(key_code, *args)

        for key_code in (mode_keys + adjust_keys + other_keys):
            self.vis.register_key_callback(key_code, make_callback(key_code))

        self.update()

    def get_pointcloud(self, frame, stride):
        pointcloud = compute_pointcloud(frame)

        normalized_frame = (frame - frame.min()) / (frame.max() - frame.min())
        colormap = cm.get_cmap(self.cmap)(normalized_frame)[..., :3].transpose((2, 0, 1))

        if stride > 1:
            pointcloud = pointcloud[:, ::stride, ::stride]
            colormap = colormap[:, ::stride, ::stride]

        pointcloud = pointcloud.reshape(3, -1)
        colormap = colormap.reshape(3, -1)

        pointcloud_colors = o3d.utility.Vector3dVector(colormap.T)
        return pointcloud, pointcloud_colors

    def update(self):
        if self.needs_redraw:
            self.needs_redraw = False

            # camera parameters are reset when adding geometry => store them
            ctr = self.vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            extrinsics = params.extrinsic

            self.vis.clear_geometries()

            data = (self.data_rotation @ self.pointcloud) + self.data_translation

            # data geometry
            if self.draw_mode == 0:
                geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.T))
                geometry.colors = self.pointcloud_colors
            else:
                geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data.T), self.triangulation)
                if self.draw_mode == 2:
                    geometry = geometry.filter_smooth_laplacian(**{'number_of_iterations': 1, 'lambda': 3})
                    geometry = geometry.filter_smooth_simple(number_of_iterations=3)
                geometry.compute_vertex_normals()

            # marker
            marker_points = o3d.utility.Vector3dVector([self.marker_position, self.marker_position + [0, 0, 300]])
            marker_geometry = o3d.geometry.LineSet(marker_points, o3d.utility.Vector2iVector([(0, 1)]))
            marker_geometry.colors = o3d.utility.Vector3dVector([(1, 0, 0)])

            self.vis.add_geometry(geometry)
            if self.show_layout:
                self.vis.add_geometry(self.layout)
            if self.show_marker:
                self.vis.add_geometry(marker_geometry)

            params.extrinsic = extrinsics
            ctr.convert_from_pinhole_camera_parameters(params)

        self.closed = not self.vis.poll_events() # returns True if window wants to close
        self.vis.update_renderer()

if __name__ == '__main__':
    frame_path = '/Volumes/Irene_CVL2/BAD-dataset/BAD1/depth/S3_file_184/img_1299.png'
    frame = imageio.imread(frame_path) / 10 # mm to cm
    layout_size = (600, 600)

    viewer = DepthViewer(frame, layout_size=layout_size, initial_extrinsic=extrinsic, cmap='winter')
    while True:
        viewer.update()