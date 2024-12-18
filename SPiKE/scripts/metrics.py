import torch
from torch import nn
import numpy as np

class MJE(nn.Module):
    def __init__(self):
        super(MJE, self).__init__()

    def forward(self, predicted, target):
        squared_distances = torch.sum((predicted - target)**2, dim=-1)
        distances = torch.sqrt(squared_distances)
        return distances.mean()


def joint_accuracy(predicted, target, threshold):

    # assert predicted.shape == target.shape == (b_size, n_kpts, 3)

    # Calculate euclidian distance between predicted and target
    distance = torch.norm(predicted - target, dim=-1)

    correct = distance < threshold
    joint_wise_correct_points = correct.sum(1).sum(0)
    frame_wise_correct_points = correct.sum(-1)

    n_frames = target.shape[0] * target.shape[1]
    n_joints = target.shape[2]

    pck = joint_wise_correct_points / n_frames * 100
    map = torch.mean(frame_wise_correct_points / n_joints) * 100
    return pck, map