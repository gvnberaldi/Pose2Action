import os
import sys
import torch
import torch.nn as nn
from typing import List

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet2_utils

class PointSpatialConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation
        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.conv_d = self._build_conv_d()
        self.mlp = self._build_mlp()

    def _build_conv_d(self):
        conv_d = [nn.Conv2d(in_channels=4, out_channels=self.mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=False)]
        if self.mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=self.mlp_planes[0]))
        if self.mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        return nn.Sequential(*conv_d)

    def _build_mlp(self):
        mlp = []
        for i in range(1, len(self.mlp_planes)):
            if self.mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=self.mlp_planes[i-1], out_channels=self.mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=False))
            if self.mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=self.mlp_planes[i]))
            if self.mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        return nn.Sequential(*mlp)

    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        device = xyzs.get_device()
        xyzs = [xyz.squeeze(dim=1).contiguous() for xyz in torch.split(xyzs, 1, dim=1)]
        new_xyzs, new_features = [], []

        for xyz in xyzs:
            reference_idx = pointnet2_utils.furthest_point_sample(xyz, xyz.size(1) // self.spatial_stride)                      # (B, N//self.spatial_stride)
            reference_xyz_flipped = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), reference_idx)           # (B, 3, N//self.spatial_stride)
            reference_xyz = reference_xyz_flipped.transpose(1, 2).contiguous()         

            idx = pointnet2_utils.ball_query(self.r, self.k, xyz, reference_xyz)
            neighbor_xyz_grouped = pointnet2_utils.grouping_operation(xyz.transpose(1, 2).contiguous(), idx)
            displacement = torch.cat((neighbor_xyz_grouped - reference_xyz_flipped.unsqueeze(3),
                                      torch.zeros((xyz.size(0), 1, xyz.size(1) // self.spatial_stride, self.k), device=device)), dim=1)
            displacement = self.conv_d(displacement)

            feature = torch.max(self.mlp(displacement), dim=-1, keepdim=False)[0]
            new_features.append(torch.max(torch.stack([feature], dim=1), dim=1, keepdim=False)[0])
            new_xyzs.append(reference_xyz)

        return torch.stack(new_xyzs, dim=1), torch.stack(new_features, dim=1)