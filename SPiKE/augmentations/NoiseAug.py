from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class NoiseAug(Augmentation):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, 
                 p_prob=1.0,
                 p_stddev=0.005,
                 p_clip=None,
                 apply_on_gt=True,
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_stddev (float): Stddev of the noise.
            p_clip (float): Maximum value to clip.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.stddev_ = p_stddev
        self.clip_ = p_clip

        # Super class init.
        super(NoiseAug, self).__init__(p_prob, apply_on_gt)

    def __compute_augmentation__(self,
                                 p_tensor,
                                 p_gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_gt_tensor (tensor): Ground Truth Tensor.
        """
        device = p_tensor.device
        cur_noise = torch.randn(
            p_tensor.shape, dtype=p_tensor.dtype).to(device)*self.stddev_
        if not self.clip_ is None:
            cur_noise = torch.clip(cur_noise, min=-self.clip_, max=self.clip_)
        aug_tensor = p_tensor + cur_noise 

        # Apply on GT tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = p_gt_tensor + cur_noise * self.stddev_
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_tensor, (cur_noise,), augmented_gt_tensor