import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.metrics import MJE

def plot_pck(pck_values, labels=None, save_path=None, title=None):

    if not labels:
        labels = np.arange(len(pck_values))

    if len(pck_values) != len(labels):
        raise Exception("The number of PCK values does not match the number of labels.")

    plt.barh(labels, pck_values, align='center')
    plt.xlabel("PCK [%]")
    plt.ylabel("Joints")
    plt.title(title)
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, len(pck_values), 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_skeletons(predicted, gt, connect_points=False, save_path=None):
    if predicted.shape != gt.shape:
        raise Exception("Predicted shape and GT shape do not match.")

    if not torch.is_tensor(predicted):
        predicted = torch.tensor(predicted)

    if not torch.is_tensor(gt):
        gt = torch.tensor(gt)

    mje = MJE()(predicted, gt)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_pred = predicted[:, 1] * -1
    y_pred = predicted[:, 0] * -1
    z_pred = predicted[:, 2]

    x_gt = gt[:, 1] * -1
    y_gt = gt[:, 0] * -1
    z_gt = gt[:, 2]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter3D(x_pred, y_pred, z_pred, c="red", label='Predicted Skeleton')
    ax.scatter3D(x_gt, y_gt, z_gt, c="blue", label='Ground Truth Skeleton')
    ax.legend(loc='upper right')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    if connect_points:
        for idx in range(gt.shape[0]):
            p = predicted[idx]
            g = gt[idx]
            ax.plot([-p[1], -g[1]], [-p[0], -g[0]], [p[2], g[2]], c='gray')

    ax.view_init(elev=90, azim=0)

    plt.title(f"Skeleton Comparison, MJE: {mje:.3f} m")

    if save_path:
        plt.savefig(save_path)

    plt.show()