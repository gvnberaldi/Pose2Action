from abc import ABC, abstractmethod
import numpy as np

class Augmentation(ABC):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, p_prob, apply_on_gt, **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used on the ground truth vector.
        """
        self.prob_ = p_prob
        self.apply_on_gt = apply_on_gt
        self.epoch_iter_ = 0

    def increase_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations.
        """
        self.epoch_iter_ += 1    

    @abstractmethod
    def __compute_augmentation__(self,
                                 p_tensor,
                                 gt_tensor=None):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            gt_tensor (tensor): (Augmented) ground truth tensor.
        """
        pass