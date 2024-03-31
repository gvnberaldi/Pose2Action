from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation
from const.skeleton_joints import flip_joint_sides

class MirrorAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob=1.0,
                 p_axes=[True, True, False],
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
        self.axes_ = p_axes

        # Super class init.
        super(MirrorAug, self).__init__(p_prob, apply_on_gt)

    def set_probs(self, probs):
        self.probs = probs

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
        
        mask_1 = torch.from_numpy(self.probs).to(device) < self.prob_
        mask_2 = torch.from_numpy(np.array(self.axes_)).to(device)
        mask = torch.logical_and(mask_1, mask_2)
        mirror_vec = torch.ones(3).to(device)*(1.-mask.to(torch.float32)) - torch.ones(3).to(device)*mask 

        aug_pts = p_pts*mirror_vec.reshape((1, -1))

        # Apply on Ground Truth tensor.
        if self.apply_on_gt:

            # Flip coordinates
            augmented_gt_tensor = p_gt_tensor * mirror_vec.reshape((1, -1))

            # If flipping in X or Z dimension, exchange positions of left and right joints
            # Flipping in both X and Z dim cancels out
            flip = torch.logical_xor(mirror_vec[0] == -1, mirror_vec[2] == -1).item()
            if flip:
                augmented_gt_tensor = flip_joint_sides(augmented_gt_tensor)

        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (mirror_vec,), augmented_gt_tensor