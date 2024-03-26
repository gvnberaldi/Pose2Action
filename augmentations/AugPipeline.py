import numpy as np
import time
import torch

from augmentations.Augmentation import Augmentation

# 3D only
from augmentations.RotationAug import RotationAug
from augmentations.ElasticDistortionAug import ElasticDistortionAug

# General.
from augmentations.CropPtsAug import CropPtsAug
from augmentations.CropBoxAug import CropBoxAug
from augmentations.MirrorAug import MirrorAug
from augmentations.TranslationAug import TranslationAug
from augmentations.NoiseAug import NoiseAug
from augmentations.DropAug import DropAug
from augmentations.LinearAug import LinearAug
from augmentations.CenterAug import CenterAug
from augmentations.STDDevNormAug import STDDevNormAug

class AugPipeline:
    """Class to represent an augmentation pipeline."""

    def __init__(self):
        """Constructor.
        """
        self.aug_classes_ = {
            sub.__name__: sub for sub in Augmentation.__subclasses__()
        }
        self.pipeline_ = []
        self.augmented_clip = []
        self.augmentation_params_list = []
        self.probabilities = []

        

    def create_pipeline(self, p_dict_list):
        """Create the pipeline.

        Args:
            p_dict_list (list of dict): List of dictionaries with the
                different augmentations in the pipeline.
        """
        self.pipeline_ = []
        for cur_dict in p_dict_list:
            self.pipeline_.append(self.aug_classes_[cur_dict['name']](**cur_dict))

        for aug in self.pipeline_:
            self.probabilities.append(torch.rand(1).item())

    def increase_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations."""
        for cur_aug in self.pipeline_:
            cur_aug.increase_epoch_counter()

    def augment(self, clip, p_extra_tensors=[]):
        """Method to augment a tensor.

        Args:
            clip (list of tensors): Input clip where each element is a tensor.
            p_extra_tensors (list): List of extra tensors.

        Return:
            aug_clip (list of tensors): Augmented clip with the same dimensions as input.
            params (list of tuples): List of parameters selected for each step of the augmentation.
            aug_extra_tensors (list): List of augmented extra tensors.
        """

        aug_clip = []
        aug_param_list = []
        prob = []

        mirror_probs = np.random.random(3)

        angle = torch.rand(1).item()


        for cur_aug in enumerate(self.pipeline_):
            prob.append(torch.rand(1).item())

            _, cur_aug_object = cur_aug 
            if isinstance(cur_aug_object, MirrorAug):
                cur_aug_object.set_probs(mirror_probs)
            elif isinstance(cur_aug_object, RotationAug):
                cur_aug_object.set_angle(angle)
            elif isinstance(cur_aug_object, LinearAug):
                if cur_aug_object.channel_independent_ and cur_aug_object.a_values_ is None:
                    cur_shape = 1
                else:
                    cur_shape = clip[0].shape[-1]
                a = torch.rand(cur_shape)
                b = torch.rand(cur_shape)
                
                cur_aug_object.set_a_b(a,b)

        for i, p_tensor in enumerate(clip):
            # Convert to PyTorch tensors if input is a NumPy array
            if isinstance(p_tensor, np.ndarray):
                cur_tensor = torch.from_numpy(p_tensor).to(torch.float32)
            else:
                cur_tensor = p_tensor.to(torch.float32)

            if isinstance(p_extra_tensors, list):
                cur_extra_tensors = [torch.from_numpy(extra).to(torch.float32) if isinstance(extra, np.ndarray) else extra.to(torch.float32) for extra in p_extra_tensors]
            else:
                cur_extra_tensors = p_extra_tensors.to(torch.float32)

            cur_aug_param_list = []

            for j, cur_aug in enumerate(self.pipeline_):
                if prob[j] <= cur_aug.prob_:
                    cur_tensor, cur_params, cur_extra_tensors = cur_aug.__compute_augmentation__(
                        cur_tensor, cur_extra_tensors)
                    cur_aug_param_list.append((cur_aug.__class__.__name__, cur_params))
            aug_clip.append(cur_tensor.numpy())
            aug_param_list.append(cur_aug_param_list)

        return torch.FloatTensor(aug_clip), aug_param_list, torch.stack(cur_extra_tensors)
