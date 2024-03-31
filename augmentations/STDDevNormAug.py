from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class STDDevNormAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self,
                 p_new_std=1.0,
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.

        Args:
            p_max_aabb_ratio (float): Maximum ratio of displacement
                wrt. the bounding box.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.stddev_ = p_new_std

        # Super class init.
        super(STDDevNormAug, self).__init__(1.0, apply_on_gt)

    def __compute_augmentation__(self,
                                 p_pts,
                                 p_gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_gt_tensor (tensor): Ground Truth Tensor.
        """
        prev_stddev = torch.amax(torch.std(p_pts, 0))
        aug_pts = (p_pts*self.stddev_)/prev_stddev

        # Apply on Ground Truth tensor
        if self.apply_on_gt:
            augmented_gt_tensor = (p_gt_tensor * self.stddev_) / prev_stddev
        else:
            augmented_gt_tensor = p_gt_tensor
        
        return aug_pts, (prev_stddev, self.stddev_), augmented_gt_tensor