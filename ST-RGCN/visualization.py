import os
import sys

from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__))
from data.dataset import SpatioTemporalGraphDataset


def plot_graph(graph):
    # Number of joints per frame
    num_joints = 15
    num_frames = 3
    # Slicing data for the first 3 frames
    num_nodes_per_frame = num_joints * num_frames
    node_features_first_3_frames = graph['joint'].x[:num_nodes_per_frame]
    # Adding 0.3 to elements 15 to 30
    node_features_first_3_frames[15:30] += 0.3
    # Adding 0.6 to elements 16 to 30
    node_features_first_3_frames[31:45] += 0.6

    spatial_edges_first_3_frames = graph['joint', 'spatial', 'joint'].edge_index[:, :num_nodes_per_frame]
    temporal_edges_first_3_frames = graph['joint', 'temporal', 'joint'].edge_index[:, :num_nodes_per_frame]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes with their coordinates as markers
    x = node_features_first_3_frames[:, 1].numpy()
    print(len(x))
    y = -node_features_first_3_frames[:, 2].numpy()
    z = node_features_first_3_frames[:, 3].numpy()

    ax.scatter(x, y, z, c='b', marker='o', label='Joints')
    cmap = plt.get_cmap('tab20')
    # Plot edges (only first 3 frames)
    for c, edge in enumerate(spatial_edges_first_3_frames.t().tolist()):
        src, dst = edge
        color = cmap(c % cmap.N)
        if src < num_nodes_per_frame and dst < num_nodes_per_frame:
            ax.plot([x[src], x[dst]], [y[src], y[dst]], [z[src], z[dst]], c=color, linewidth=4, linestyle='-',
                    alpha=0.5)

    for edge in temporal_edges_first_3_frames.t().tolist():
        src, dst = edge
        if src < num_nodes_per_frame and dst < num_nodes_per_frame:
            ax.plot([x[src], x[dst]], [y[src], y[dst]], [z[src], z[dst]], c='r', linestyle='--', linewidth=2, alpha=0.5)

    # Manually add legend entry for temporal edge
    ax.plot([], [], c='r', linestyle='--', linewidth=2, label='Temporal Edge')

    # Set plot labels and legend
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    # Adjust axis limits to make the point cloud smaller
    ax.set_xlim([-0.5, 2])
    ax.set_ylim([-1.7, 1.3])

    # Adjust viewpoint
    ax.view_init(elev=90, azim=90)
    ax.axis('off')

    plt.title('Spatio-Temporal Graph Visualization')
    plt.show()


if __name__ == "__main__":
    # Example usage
    dataset = SpatioTemporalGraphDataset(root=os.path.join(os.getcwd(), 'dataset\\BAD'),
                                         activity_label_file=os.path.join(os.getcwd(), 'dataset\\activity_labels.txt'))

    # Access individual graphs
    for idx in range(len(dataset)):
        graph = dataset[idx]
        print(f"Graph {idx}: {graph}")

    plot_graph(dataset.get_hetero_data(0))

    '''
    for idx in range(len(dataset)):
        hetero_graph = dataset.get_hetero_data(idx)
        plot_graph(hetero_graph)  # Plot the graph
    '''