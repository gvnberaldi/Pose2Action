import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

#P4T
from point_4d_convolution import *
from transformer import *

#PSTNet
from pst_convolutions import PSTConv

#PSTNet2
from pst_operations import PSTOp

#PSTT
from transformer_v1 import TransformerPSTT

#PPTr
from models.PPTr_pointnet import pointnet
from transformer_pptr import TransformerPPTr

class HPETransformer(nn.Module):
    def __init__(self, num_points, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[0, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        print("dim ", dim)
        print("dim_head ", dim_head)
        print(num_points/spatial_stride*dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_points//spatial_stride*dim),
            nn.Linear(num_points//spatial_stride*dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]


        features = features.permute(0, 1, 3, 2)
                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)

        #flatten the output tensor without any pooling
        output = output.view(output.size(0), -1)


        #output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        
        output = self.mlp_head(output)

        return output

class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output

    def freeze_transformer_layers(self, num_layers_to_unfreeze):
        """
        Freeze the specified number of transformer layers starting from the last layer.

        Args:
            num_layers_to_unfreeze (int): Number of transformer layers to unfreeze.
        """
        # Determine the starting index for freezing
        start_idx = len(self.transformer.layers) - num_layers_to_unfreeze

        # Freeze the parameters of layers up to the specified index
        for i in range(start_idx):
            for param in self.transformer.layers[i].parameters():
                param.requires_grad = False

        # Unfreeze the parameters of the specified number of layers
        for i in range(start_idx, len(self.transformer.layers)):
            for param in self.transformer.layers[i].parameters():
                param.requires_grad = True

class PSTNet(nn.Module):
    def __init__(self, radius=1.5, nsamples=3*3, num_classes=20):
        super(PSTNet, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=284,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out
    
class PSTNet2(nn.Module):
    def __init__(self, radius=0.5, nsamples=3*3, num_classes=20):
        super(PSTNet2, self).__init__()

        self.encoder1  = PSTOp(in_channels=0,
                               spatial_radius=radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[45],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_channels=[64],
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder2a = PSTOp(in_channels=64,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[96],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_padding=[1,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[128],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder2b = PSTOp(in_channels=128,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[192],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[1,1],
                               temporal_padding_mode="replicate",
                               temporal_channels=[256],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder3a = PSTOp(in_channels=256,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[384],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_padding=[1,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[512],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder3b = PSTOp(in_channels=512,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[768],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[1,1],
                               temporal_padding_mode="replicate",
                               temporal_channels=[1024],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder4  = PSTOp(in_channels=1024,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[1536],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[2048],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.encoder1(xyzs, None)

        new_xys, new_features = self.encoder2a(new_xys, new_features)

        new_xys, new_features = self.encoder2b(new_xys, new_features)

        new_xys, new_features = self.encoder3a(new_xys, new_features)

        new_xys, new_features = self.encoder3b(new_xys, new_features)

        _, new_features = self.encoder4(new_xys, new_features)                      # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)        # (B, L, C)
        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]        # (B, C)

        out = self.fc(new_feature)

        return out

class PSTTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = TransformerPSTT(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        features = features.permute(0, 1, 3, 2)

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output
    
class PrimitiveTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = TransformerPPTr(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = TransformerPPTr(dim, depth, heads, dim_head, mlp_dim)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):


        # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        point_feat = torch.reshape(input=raw_feat, shape=(B * L * 8, -1, C))  # [B*L*4, n', C]
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 8, C))  # [B, L*4, C]



        primitive_feature = self.emb_relu2(primitive_feature)
        primitive_feature = self.transformer2(primitive_feature) # B. L*4, C

        output = torch.max(input=primitive_feature, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output