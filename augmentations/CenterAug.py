from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class CenterAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self,
                 p_prob=1.0,
                 p_axes=[True, True, True],
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.
        """

        self.axes_ = p_axes

        # Super class init.
        super(CenterAug, self).__init__(p_prob, apply_on_gt)

    def __compute_augmentation__(self,
                                 p_pts,
                                 p_gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_gt_tensor (tensor): Ground truth tensor.
        """

        # Center
        axes_mask = np.logical_not(np.array(self.axes_))
        center_pt = torch.mean(p_pts, 0)
        aug_pts = p_pts - center_pt.reshape((1, -1))
        aug_pts[:, axes_mask] = p_pts[:, axes_mask]

        # Apply on GT tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor - center_pt.reshape((1, -1))
            augmented_gt_tensor[:,axes_mask] = p_gt_tensor[:,axes_mask]
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (center_pt, axes_mask), augmented_gt_tensor