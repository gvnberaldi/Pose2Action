from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class DropAug(Augmentation):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, 
                 p_prob=1.0,
                 apply_on_gt=True,
                 p_drop_prob=0.05,
                 p_keep_zeros=True,
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_apply_extra_tensors (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
            p_prob (float): Remove probability.
            p_keep_zeros (bool): Boolean that indicates if the 
                drop values are kept as zeros.
        """

        # Store variables.
        self.drop_prob_ = p_drop_prob
        self.keep_zeros_ = p_keep_zeros

        # Super class init.
        super(DropAug, self).__init__(p_prob, apply_on_gt)

    def __compute_augmentation__(self,
                                 p_tensor,
                                 p_gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_gt_tensor (tensor): Ground Truth tensor.
        """
        device = p_tensor.device
        mask = torch.rand(p_tensor[:, 0].shape).to(p_tensor.device) > self.drop_prob_
        
        if self.keep_zeros_:
            aug_tensor = p_tensor*mask + torch.ones_like(p_tensor)*(1.-mask)

            if self.apply_on_gt:
                augmented_gt_tensor = p_gt_tensor * mask + torch.ones_like(p_gt_tensor) * (1. - mask)
            else:
                augmented_gt_tensor = p_gt_tensor

        else:
            aug_tensor = p_tensor[mask]

            if self.apply_on_gt:
                augmented_gt_tensor = p_gt_tensor[mask]
            else:
                augmented_gt_tensor = p_gt_tensor

        return aug_tensor, (mask, ), augmented_gt_tensor