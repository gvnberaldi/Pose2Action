import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Compute the standard Cross Entropy Loss
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Compute the weights using alpha and target
        weights = torch.pow((1 - F.softmax(input, dim=1).gather(1, target.view(-1, 1))).squeeze(), self.gamma)
        weights = self.alpha * weights

        # Apply the weights to the standard Cross Entropy Loss
        focal_loss = ce_loss * weights

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss