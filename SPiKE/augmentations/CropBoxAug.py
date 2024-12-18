from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class CropBoxAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob=1.0,
                 p_min_crop_size=0.5,
                 p_max_crop_size=1.0,
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_min_crop_size (float): Minimum crop size.
            p_max_crop_size (float): Maximum crop size.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.min_crop_size_ = p_min_crop_size
        self.max_crop_size_ = p_max_crop_size

        # Super class init.
        super(CropBoxAug, self).__init__(p_prob, apply_on_gt)

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
        min_pt = torch.amin(p_pts, 0)
        max_pt = torch.amax(p_pts, 0)
        aabb_size = max_pt - min_pt

        valid_augment = False

        while not valid_augment:

            cur_crop_size = torch.rand(p_pts.shape[-1]).to(device)*\
                (self.max_crop_size_ - self.min_crop_size_) + self.min_crop_size_
            cur_crop_size = torch.minimum(cur_crop_size, aabb_size)

            limits_box = max_pt - cur_crop_size
            cur_crop_point = torch.rand(p_pts.shape[-1]).to(device)*\
                (limits_box - min_pt) + min_pt

            mask = torch.logical_and(p_pts[:, 0] >= cur_crop_point[0], p_pts[:, 0] <= (cur_crop_point[0] + cur_crop_size[0]))
            for i in range(1, p_pts.shape[-1]):
                mask = torch.logical_and(mask, 
                    torch.logical_and(p_pts[:, i] >= cur_crop_point[i], p_pts[:, i] <= (cur_crop_point[i] + cur_crop_size[i])))
                
            aug_pts = p_pts[mask]
            valid_augment = aug_pts.shape[0] > 0

        # Apply on GT tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor[mask]
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (mask, cur_crop_point, cur_crop_size), augmented_gt_tensor