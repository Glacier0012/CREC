# coding=utf-8

from .lr_scheduler import StepLR, WarmupCosineLR

def build_lr_scheduler(cfg, optimizer, n_iter_per_epoch):
    """Build learning rate scheduler."""
    scheduler_name = cfg.train.scheduler.name.lower()
    
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = WarmupCosineLR(
            optimizer=optimizer,
            warmup_epochs=cfg.train.warmup_epochs,
            epochs=cfg.train.epochs,
            warmup_lr=cfg.train.warmup_lr,
            base_lr=cfg.train.base_lr,
            min_lr=cfg.train.min_lr,
            n_iter_per_epoch=n_iter_per_epoch
        )
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer=optimizer,
            warmup_epochs=cfg.train.warmup_epochs,
            epochs=cfg.train.epochs,
            decay_epochs=cfg.train.scheduler.decay_epochs,
            lr_decay_rate=cfg.train.scheduler.lr_decay_rate,
            n_iter_per_epoch=n_iter_per_epoch,
        )
    
    return scheduler
