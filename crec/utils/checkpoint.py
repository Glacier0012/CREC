# coding=utf-8

import os

import torch

from crec.utils.distributed import is_main_process


def load_checkpoint(cfg, model, optimizer, scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.train.resume_path}....................")
    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda())
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    #scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    logger.info("==> loaded checkpoint from {}\n".format(cfg.train.resume_path) +
                "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))
    return start_epoch + 1


def save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, det_best=False, seg_best=False, normal=True):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'lr': optimizer.param_groups[0]["lr"]
    }
    if normal:
        save_path = os.path.join(cfg.train.output_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(save_state, save_path)

    # save last checkpoint
    last_checkpoint_path = os.path.join(cfg.train.output_dir, f'last_checkpoint.pth')
    torch.save(save_state, last_checkpoint_path)

    # save best detection model
    if det_best:
        det_best_model_path = os.path.join(cfg.train.output_dir, f'det_best_model.pth')
        torch.save(save_state, det_best_model_path)
    
    # save best cf_acc model
    if seg_best:
        seg_best_model_path = os.path.join(cfg.train.output_dir, f'cf_best_model.pth')
        torch.save(save_state, seg_best_model_path)


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        resume_file = os.path.join(output_dir, "last_checkpoint.pth")
    else:
        resume_file = None

    return resume_file