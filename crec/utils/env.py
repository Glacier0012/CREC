# coding=utf-8

import os
import random
import warnings
import numpy as np
from typing import Optional

import torch
import torch.backends.cudnn as cudnn


def seed_everything(SEED: Optional[int]):
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(SEED)

        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def setup_unique_version(cfg):
    while True:
        version = random.randint(0, 99999)
        if not (os.path.exists(os.path.join(cfg.train.log_path ,str(version)))):
            cfg.train.version = str(version)
            break