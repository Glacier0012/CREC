import os
import time
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crec.config import instantiate, LazyConfig
from crec.datasets.dataloader import build_test_loader
from crec.datasets.utils import yolobox2label
from crec.models.utils import batch_box_iou
from crec.utils.env import seed_everything
from crec.utils.logger import create_logger
from crec.utils.metric import AverageMeter
from crec.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, writer, epoch, ix_to_token, logger, rank, save_ids=None, prefix='Val', ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU_0.5', ':6.2f')
    cls_acc = AverageMeter('Cls_acc', ':6.2f')
    cf_acc = AverageMeter('Cf_acc', ':6.2f')
    meters = [batch_time, data_time, losses, box_ap, cls_acc, cf_acc]
    meters_dict = {meter.name: meter for meter in meters}
    
    with torch.no_grad():
        end = time.time()
        confusion_matrix = torch.zeros([2,2])
        for idx, (ref_iter, image_iter, box_iter, gt_box_iter, info_iter, aw_iter, negs_iter) in enumerate(data_loader):
            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)
            aw_iter = aw_iter.cuda(non_blocking=True) if aw_iter.shape[-1] > 1 else None
            negs_iter = negs_iter.cuda(non_blocking=True) if negs_iter.shape[-1] > 1 else None
            
            box, cls_pred = model(image_iter, ref_iter, aw_iter, negs_iter)
            
            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box=box.squeeze(1).cpu().numpy()

            cls_predictions = torch.argmax(cls_pred, dim=1)
            cf_label = box_iter[:,:,-1].view(-1)

            for p, gt in zip(cls_predictions.cpu().numpy().astype(np.int8), cf_label.cpu().numpy().astype(np.int8)):
                confusion_matrix[p, gt] += 1.0
            
            # predictions to ground-truth
            for i in range(len(gt_box_iter)):
                box[i] = yolobox2label(box[i], info_iter[i])

            # take the positives
            TP_idx = []
            pos_idx = []
            TN_count = 0
            for id in range(cf_label.shape[0]):
                if cf_label[id] == 0:
                    pos_idx.append(id)
                    if cls_predictions[id] == 0:    # normal postive sample
                        TP_idx.append(id)
                else: 
                    if cls_predictions[id] == 1:    # counterfactual postive sample
                        TN_count+=1

            if len(pos_idx) > 0:
                box_iou=batch_box_iou(torch.from_numpy(gt_box_iter[pos_idx]),torch.from_numpy(box[pos_idx])).cpu().numpy()
                box_ap.update((box_iou>0.5).astype(np.float32).mean(), box_iou.shape[0])
                tp_box_iou=batch_box_iou(torch.from_numpy(gt_box_iter[TP_idx]),torch.from_numpy(box[TP_idx])).cpu().numpy()                
                cf_acc.update(((tp_box_iou>0.5).sum()+TN_count)/cf_label.shape[0], cf_label.shape[0])
            else:
                cf_acc.update(TN_count/cf_label.shape[0], cf_label.shape[0])
            
            cls_acc.update((cls_predictions == cf_label).sum()/cf_label.shape[0], cls_predictions.shape[0])
            reduce_meters(meters_dict, rank, cfg)

            if (idx % cfg.train.log_period == 0 or idx==(len(data_loader)-1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'BoxIoU_0.5 {box_ap.val:.4f} ({box_ap.avg:.4f})  '
                    f'Cls_Acc {cls_acc.val:.4f} ({cls_acc.avg:.4f})  '
                    f'Cf_Acc {cf_acc.val:.4f} ({cf_acc.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            batch_time.update(time.time() - end)
            end = time.time()

        if is_main_process() and writer is not None:
            writer.add_scalar("Acc/BoxIoU_0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/Cls_Acc", cls_acc.avg_reduce, global_step=epoch)

        logger.info(f' * BoxIoU_0.5: {box_ap.avg_reduce:.3f} Cls_Acc: {cls_acc.avg_reduce:.3f} Cf_Acc: {cf_acc.avg_reduce:.3f}')
        logger.info(f' * Confusion_matrix: {confusion_matrix.data}')

    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, cls_acc.avg_reduce


def main(cfg):
    global best_det_acc
    best_det_acc=0.

    # build single or multi-datasets for validation
    loaders=[]
    prefixs = []
    val_set = instantiate(cfg.dataset)
    prefixs=['val']
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader=build_test_loader(cfg, val_set, shuffle=False, drop_last=False)
    loaders.append(val_loader)
    
    if cfg.dataset.dataset in ['refcoco', 'refcoco+', 'c-refcoco', 'c-refcoco+']:
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_test_loader(cfg, testA_dataset, shuffle=False, drop_last=False)

        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        testB_loader = build_test_loader(cfg, testB_dataset, shuffle=False, drop_last=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    else:                               # for refcocog and c-refcocog
        cfg.dataset.split = "test"
        test_dataset=instantiate(cfg.dataset)
        test_loader=build_test_loader(cfg, test_dataset, shuffle=False, drop_last=False)
        prefixs.append('test')
        loaders.append(test_loader)
    
    # build model
    cfg.model.language_encoder.pretrained_emb = val_set.pretrained_emb
    cfg.model.language_encoder.token_size = val_set.token_size
    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    torch.cuda.set_device(dist.get_rank())
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda() )
    model_without_ddp.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if cfg.train.amp:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None

    save_ids=np.random.randint(1, len(testA_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None
    for data_loader, prefix in zip(loaders, prefixs):
        box_ap, cls_acc = validate(
            cfg=cfg, 
            model=model, 
            data_loader=data_loader, 
            writer=writer, 
            epoch=0, 
            ix_to_token=testA_dataset.ix_to_token,
            logger=logger,
            rank=dist.get_rank(),
            save_ids=save_ids,
            prefix=prefix)
        logger.info(f' * BoxIoU_0.5 {box_ap:.3f} Cls_Acc {cls_acc:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CREC")
    parser.add_argument('--config', type=str, required=True, default='./config/crec_refcoco.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Environments setting
    seed_everything(cfg.train.seed)

    # Distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend, 
        init_method=cfg.train.ddp.init_method, 
        world_size=world_size, 
        rank=rank
    )
    torch.distributed.barrier()

    # Path setting
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    # Refine cfg for evaluation
    cfg.train.resume_path = args.eval_weights
    logger.info(f"Running evaluation from specific checkpoint {cfg.train.resume_path}......")

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
