from __future__ import print_function
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import errno
import os

import wandb

from collections import Counter

import tqdm

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def compute_class_weights(data_loader, num_classes):
    # Initialize a Counter to count class occurrences
    class_counts = Counter()

    for _, targets, _, _ in data_loader:
        class_counts.update(targets.tolist())
    
    # Convert class_counts to a list of counts
    class_counts_list = [class_counts[i] for i in range(0, num_classes)]

    print(class_counts_list)

    # Calculate class weights, considering zero counts
    total_samples = sum(class_counts_list)
    class_weights = []

    for count in class_counts_list:
        if count == 0:
            class_weights.append(0.0)  # Assign zero weight if count is zero
        else:
            class_weights.append(total_samples / (num_classes * count))

    print("Class Weights:", class_weights)

    return class_weights

def count_parameters(model: nn.Module) -> int:
    """
    Counts parameters for training in a given model.
    Args:
        model (nn.Module): Input model
    Returns:
        int: No. of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False