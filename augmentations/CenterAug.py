from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class CenterAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self,
                 p_axes = [True, True, True],
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor.
        """

        self.axes_ = p_axes

        # Super class init.
        super(CenterAug, self).__init__(1.0, p_apply_extra_tensors)


    def __compute_augmentation__(self,
                                 p_pts,
                                 p_extra_tensors = []):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_extra_tensors (list): List of extra tensors.
        """

        # Center
        axes_mask = np.logical_not(np.array(self.axes_))
        center_pt = torch.mean(p_pts, 0)
        aug_pts = p_pts - center_pt.reshape((1, -1))
        aug_pts[:,axes_mask] = p_pts[:,axes_mask]

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_tensor = cur_tensor - center_pt.reshape((1, -1))
                new_tensor[:,axes_mask] = cur_tensor[:,axes_mask]
                new_extra_tensors.append(new_tensor)
            else:
                new_extra_tensors.append(cur_tensor)


        return aug_pts, (center_pt, axes_mask), new_extra_tensors
