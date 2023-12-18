from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class LinearAug(Augmentation):
    """Class to represent a elastic distortion point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_min_a = 0.9,
                 p_max_a = 1.1,
                 p_min_b = -0.1,
                 p_max_b = 0.1,
                 p_a_values = None,
                 p_b_values = None,
                 p_channel_independent = False, 
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor. Augmentation y = x*a + b

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_min_a (float): Minimum a.
            p_max_a (float): Maximum a.
            p_min_b (float): Minimum b.
            p_max_b (float): Maximum b.
            p_a_values (list of floats): List of user defined a values.
            p_b_values (list of floats): List of user defined b values.
            p_channel_independent (bool): Boolean that indicates if
                the transformation is applied independent of the channel.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.min_a_ = p_min_a
        self.max_a_ = p_max_a
        self.min_b_ = p_min_b
        self.max_b_ = p_max_b
        self.a_values_ = p_a_values
        self.b_values_ = p_b_values
        self.channel_independent_ = p_channel_independent
        self.a = None
        self.b = None

        # Super class init.
        super(LinearAug, self).__init__(p_prob, p_apply_extra_tensors)

    def set_a_b(self,a,b):
        self.a = a
        self.b = b

    def __compute_augmentation__(self,
                                 p_tensor,
                                 p_extra_tensors = []):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_extra_tensors (list): List of extra tensors.
        """
        device = p_tensor.device
        if self.channel_independent_ and self.a_values_ is None:
            cur_shape = 1
        else:
            cur_shape = p_tensor.shape[-1]
        
        if self.a_values_ is None:
            cur_a = self.a.to(device)*\
                (self.max_a_ - self.min_a_) + self.min_a_
            cur_b = self.b.to(device)*\
                (self.max_b_ - self.min_b_) + self.min_b_
        else:
            cur_a = torch.from_numpy(np.array(self.a_values_[self.epoch_iter_])).to(device)
            cur_b = torch.from_numpy(np.array(self.b_values_[self.epoch_iter_])).to(device)
        
        aug_tensor = p_tensor*cur_a.reshape((1, -1)) + cur_b.reshape((1, -1))

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    cur_tensor*cur_a.reshape((1, -1)) + cur_b.reshape((1, -1)))
            else:
                new_extra_tensors.append(cur_tensor)

        return aug_tensor, (cur_a,cur_b), new_extra_tensors
