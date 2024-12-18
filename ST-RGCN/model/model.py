import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_dim, num_classes, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, mlp_dim),  # Adjust hidden layer size as needed
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_type, batch):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Use global pooling (e.g., mean pooling) to get a graph-level representation

        # Predict using MLP
        x_mlp = self.mlp(x)

        return F.log_softmax(x_mlp, dim=1)
