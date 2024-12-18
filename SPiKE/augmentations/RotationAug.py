from abc import ABC, abstractmethod
import numpy as np
import torch

from augmentations.Augmentation import Augmentation

class RotationAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_axis = 0,
                 p_min_angle= 0,
                 p_max_angle = 2*np.pi,
                 p_angle_values = None,
                 apply_on_gt=True,
                 **kwargs):
        """Constructor. 

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axis (Int): Index of the axis.
            p_min_angle (float): Minimum rotation angle.
            p_max_angle (float): Maximum rotation angle.
            p_angle_values (list of floats): User-defined angle per epoch.
            apply_on_gt (bool): Boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.axis_ = p_axis
        self.min_angle_ = p_min_angle
        self.max_angle_ = p_max_angle
        self.angle_values_ = p_angle_values
        self.cur_angle = None

        # Super class init.
        super(RotationAug, self).__init__(p_prob, apply_on_gt)
    
    def set_angle(self, angle):

        if self.angle_values_ is None:
            self.cur_angle = angle*\
                (self.max_angle_-self.min_angle_) + self.min_angle_
        else:
            self.cur_angle = self.angle_values_[self.epoch_iter_]

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
        if self.axis_ == 0:
            R = torch.from_numpy(
                np.array([[1.0, 0.0, 0.0],
                          [0.0, np.cos(self.cur_angle), -np.sin(self.cur_angle)],
                          [0.0, np.sin(self.cur_angle), np.cos(self.cur_angle)]])).to(device).to(torch.float32)
        elif self.axis_ == 1:
            R = torch.from_numpy(
                np.array([[np.cos(self.cur_angle), 0.0, np.sin(self.cur_angle)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(self.cur_angle), 0.0, np.cos(self.cur_angle)]])).to(device).to(torch.float32)
        elif self.axis_ == 2:
            R = torch.from_numpy(
                np.array([[np.cos(self.cur_angle), -np.sin(self.cur_angle), 0.0],
                          [np.sin(self.cur_angle), np.cos(self.cur_angle), 0.0],
                          [0.0, 0.0, 1.0]])).to(device).to(torch.float32)

        aug_pts = torch.matmul(p_pts, R)

        # Apply on GT tensor.
        if self.apply_on_gt:
            augmented_gt_tensor = torch.matmul(p_gt_tensor, R)
        else:
            augmented_gt_tensor = p_gt_tensor

        return aug_pts, (self.axis_, self.cur_angle), augmented_gt_tensor