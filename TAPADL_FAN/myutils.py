"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import cv2
from PIL import Image

import torch
import torch.distributed as dist
import numpy as np
from torchvision import datasets, transforms

import torch.nn.functional as F
from timm.utils import *


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.2f} ({global_avg:.2f})"
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


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


class MyCheckpointSaver(CheckpointSaver):
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        super().__init__(model,
                         optimizer,
                         args,
                         model_ema,
                         amp_scaler,
                         checkpoint_prefix,
                         recovery_prefix,
                         checkpoint_dir,
                         recovery_dir,
                         decreasing,
                         max_history,
                         unwrap_fn)

    def load_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_load = torch.load(tmp_save_path)
        self.checkpoint_files = file_to_load['checkpoint_files']
        self.best_metric = file_to_load['best_metric']

    def save_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_save = {'checkpoint_files': self.checkpoint_files, 'best_metric': self.best_metric}
        torch.save(file_to_save, tmp_save_path)


