# coding=utf-8

import math

import torch
import torch.optim.lr_scheduler as lr_scheduler


def StepLR(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    epochs: int,
    decay_epochs: int,
    lr_decay_rate: float,
    n_iter_per_epoch: int,
):
    t, T = warmup_epochs * n_iter_per_epoch, epochs * n_iter_per_epoch
    def lr_func(step):
                coef = 1.
                if step<=t:
                    coef=float(step)/float(t+1)
                else:
                    for i,deps in enumerate(decay_epochs):
                        if step>=deps * n_iter_per_epoch:
                            coef=lr_decay_rate**(i+1)
                return coef
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


def CosineDecayLR(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    n_iter_per_epoch: int,
):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * n_iter_per_epoch)
    return scheduler


def WarmupCosineLR(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    epochs: int,
    warmup_lr: float,
    base_lr: float,
    min_lr: float,
    n_iter_per_epoch: int,
):
    t, T = warmup_epochs * n_iter_per_epoch, epochs * n_iter_per_epoch
    n_t=0.5
    warm_step_lr=(base_lr - warmup_lr) / t
    #print("--------",base_lr, warmup_lr, warm_step_lr, '-------------')
    lr_func = lambda step: ( step*warm_step_lr + warmup_lr) / base_lr if step < t \
        else (min_lr + n_t * (base_lr - min_lr) * (1 + math.cos(math.pi * (step - t) / (T - t)))) / base_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler