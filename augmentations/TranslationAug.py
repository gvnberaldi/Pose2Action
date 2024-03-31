from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class TranslationAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_max_aabb_ratio = 1.0, 
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_max_aabb_ratio (float): Maximum ratio of displacement
                wrt. the bounding box.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.max_aabb_ratio_ = p_max_aabb_ratio

        # Super class init.
        super(TranslationAug, self).__init__(p_prob, apply_on_gt)

    def __compute_augmentation__(self,
                                 p_pts,
                                 p_gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_gt_tensor (tensor): Ground Truth tensor.
        """
        device = p_pts.device
        cur_translation = (torch.rand(p_pts.shape[-1]).to(device)*2. - 1.)*self.max_aabb_ratio_
        min_pt = torch.amin(p_pts, 0)
        max_pt = torch.amax(p_pts, 0)
        displacement_vec = (max_pt - min_pt)/2. * cur_translation
        aug_pts = p_pts + displacement_vec.reshape((1, -1))

        # Apply on ground truth tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor + displacement_vec.reshape((1, -1))
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (displacement_vec,), augmented_gt_tensor