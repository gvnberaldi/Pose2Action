import os

import h5py
import torch
from torch_geometric.data import HeteroData, Dataset


class SpatioTemporalGraphDataset(Dataset):
    def __init__(self, root, activity_label_file):
        super(SpatioTemporalGraphDataset, self).__init__()
        self.data, self.hetero_data = self._process_data(activity_label_file, root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_hetero_data(self, idx):
        return self.hetero_data[idx]

    def _process_data(self, activity_label_file, root):
        activity_data = self._process_activity_label_file(activity_label_file)
        skeletons_and_labels = self._get_skeleton_and_labels(activity_data, root)
        graphs, hetero_graphs = zip(*[self._create_spatio_temporal_graph(skeleton, label)
                                      for skeleton, label in skeletons_and_labels])
        graphs = list(graphs)
        hetero_graphs = list(hetero_graphs)
        return graphs, hetero_graphs

    def _process_activity_label_file(self, activity_label_file):
        with open(activity_label_file, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            parts = line.strip().split('\t')
            file_number = parts[0]
            subject = parts[1]
            class_label = parts[2]
            start_frame = int(parts[3])
            end_frame = int(parts[4])
            data.append((file_number, subject, class_label, start_frame, end_frame))

        return data

    def _get_skeleton_and_labels(self, data, root):
        result = []
        for file_number, subject, class_label, start_frame, end_frame in data:
            # Skip if class label is 10 or 12
            if class_label in {'10', '12'}:
                # print(f"Skipping class label {class_label} for {subject}_{file_number}.")
                continue
            dataset_path = os.path.join(root, f'{subject}_{file_number}_labels.h5')
            # Check if the dataset file exists
            if not os.path.exists(dataset_path):
                # print(f"Dataset file {dataset_path} does not exist. Skipping.")
                continue
            with h5py.File(dataset_path, 'r') as f:
                ids = f['id'][:]
                skeletons_data = f['3d_joints_coordinates'][:]
            frame_numbers = {f"{frame:05d}" for frame in range(start_frame, end_frame + 1)}
            indices = [i for i, id in enumerate(ids) if id.decode('utf-8').split('_')[1] in frame_numbers]
            # Check if indices are not empty
            if not indices:
                # print(f"No valid indices found for {dataset_path}. Skipping.")
                continue
            skeletons = torch.tensor(skeletons_data[indices])
            result.append((skeletons, class_label))
        return result

    def _create_spatio_temporal_graph(self, skeleton_data, label):
        num_frames, num_joints, num_features = skeleton_data.shape

        # Define the skeleton connectivity
        skeleton_edges = [
            (0, 1), (1, 2), (1, 3), (1, 8), (2, 4), (3, 5),
            (4, 6), (5, 7), (8, 9), (8, 10), (9, 11), (10, 12),
            (11, 13), (12, 14),
        ]

        # Initialize lists to store edge indices
        spatial_edges = []
        temporal_edges = []

        # Add spatial edges for each frame
        for frame in range(num_frames):
            for src, dst in skeleton_edges:
                spatial_edges.append((frame * num_joints + src, frame * num_joints + dst))

        # Add temporal edges for each joint across consecutive frames
        for frame in range(num_frames - 1):
            for joint in range(num_joints):
                temporal_edges.append((frame * num_joints + joint, (frame + 1) * num_joints + joint))

        # Convert edge lists to tensors
        spatial_edges = torch.tensor(spatial_edges, dtype=torch.long).t().contiguous()
        temporal_edges = torch.tensor(temporal_edges, dtype=torch.long).t().contiguous()

        # Create node features
        node_features = skeleton_data.reshape(-1, num_features).float()  # shape: (num_frames * num_joints, num_features)

        num_anatomic_ids = 15  # From 0 to 14
        total_nodes = node_features.shape[0]

        # Create cyclic anatomic IDs
        anatomic_ids = torch.arange(num_anatomic_ids)
        anatomic_ids = anatomic_ids.repeat(total_nodes // num_anatomic_ids)

        # Append the anatomic IDs as an additional feature
        anatomic_ids = anatomic_ids.view(-1, 1).float()
        node_features = torch.cat((anatomic_ids, node_features), dim=1)  # shape (num_frames * num_joints, num_features + 1)

        # Create a HeteroData object
        hetero_data = HeteroData()

        # Add node features
        hetero_data['joint'].x = node_features
        # Add spatial edges
        hetero_data['joint', 'spatial', 'joint'].edge_index = spatial_edges
        # Add temporal edges
        hetero_data['joint', 'temporal', 'joint'].edge_index = temporal_edges

        hetero_data['label'] = torch.tensor(int(label), dtype=torch.long)

        # Convert to homogeneous graph
        data = hetero_data.to_homogeneous(add_node_type=False)

        return data, hetero_data
