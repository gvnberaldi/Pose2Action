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

        self.displacement_vec = None

    def set_displacement(self, displacement_vec):
        # Method to set the displacement vector externally
        self.displacement_vec = displacement_vec


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

        # Use the pre-set displacement vector
        self.displacement_vec = self.displacement_vec.to(device)

        aug_pts = p_pts + self.displacement_vec.reshape((1, -1))

        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor + self.displacement_vec.reshape((1, -1))
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (self.displacement_vec,), augmented_gt_tensor