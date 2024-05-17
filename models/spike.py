import torch
from torch import nn
import sys 
import os

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
from point_spat_convolution import *
from transformer import *

class SPiKE(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride, dim, depth, heads, dim_head, mlp_dim, num_coord_joints, dropout1=0.0, dropout2=0.0):
        super().__init__()

        self.stem = PointSpatialConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                     spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride)

        self.pos_embed = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_coord_joints),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.stem(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        B, L, n, _ = xyzs.shape
        t = torch.arange(L, device=device).view(1, L, 1, 1).expand(B, -1, n, -1) + 1
        xyzts = torch.cat((xyzs, t), dim=-1)
        xyzts = xyzts.view(B, -1, 4)  # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)  # [B, L, n, C]
        features = features.reshape(B, -1, features.shape[3])  # [B, L*n, C]
        
        xyzts = self.pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        joints_coord = self.mlp_head(output)

        return joints_coord
    
