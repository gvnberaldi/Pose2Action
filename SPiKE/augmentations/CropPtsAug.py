from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class CropPtsAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob=1.0,
                 p_max_pts=0,
                 p_crop_ratio=1.0,
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_max_pts (float): Maximum number of points.
            p_crop_ratio (float): Scene crop ratio.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.max_pts_ = p_max_pts
        self.crop_ratio_ = p_crop_ratio

        # Super class init.
        super(CropPtsAug, self).__init__(p_prob, apply_on_gt)

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
        device = p_pts.device
        cur_num_pts = p_pts.shape[0]
        max_num_pts = self.max_pts_ if self.max_pts_ > 0 else cur_num_pts
        max_num_pts = min(max_num_pts, int(cur_num_pts * self.crop_ratio_))

        crop_mask = torch.ones(cur_num_pts, dtype=torch.bool).to(p_pts.device)
        if cur_num_pts > max_num_pts:
            rand_idx = torch.randint(low=0, high=cur_num_pts, size=(1,)).to(p_pts.device)
            sort_idx = torch.argsort(torch.sum((p_pts - p_pts[rand_idx])**2, 1))
            crop_idx = sort_idx[max_num_pts:]
            crop_mask[crop_idx] = False
            aug_pts = p_pts[crop_mask]

        # Ground Truth tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor[crop_mask]
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, crop_mask, augmented_gt_tensor