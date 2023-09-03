"""
Train and eval functions used in main.py
"""
import os
import math
import sys
from typing import Iterable, Optional
import copy

import torch
import torchvision
from torchvision import datasets, transforms
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from losses import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import build_dataset
from torchvision.utils import save_image
from timm.models.layers import trunc_normal_, DropPath
from autoattack import AutoAttack


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    if H==224:
        ps = 16
    elif H==32:
        ps = 4
    patches = input1.unfold(2, ps, ps).unfold(3, ps, ps).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,ps,ps)
    patches = patch_transform(patches)

    patches = patches.reshape(bs, -1, channle_size,ps,ps).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*ps*ps, -1)
    output_images = F.fold(patches, (H,W), ps, stride=ps)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images


def train_one_epoch(logger, args, model: torch.nn.Module, criterion: DistillationLoss, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, set_training_mode=True, controller_updater=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        acc_targets = targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            _, _, H, W = samples.shape
            if H==224:
                ps = 16
            elif H==32:
                ps = 4
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(ps,ps), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
            )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast(enabled=False):
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)

            # forward for normal data
            outputs, attns = model(samples, return_attn=True)
            logits = outputs
            # compute loss
            loss = criterion(samples, outputs, targets)

            # Attention Diversification Loss
            attn_div_loss = AttentionDiversificationLoss(attns, args.threshold)
            loss = loss + args.adloss * attn_div_loss

        loss_value = loss.item()

        acc1, acc5 = accuracy(logits, acc_targets, topk=(1, 5))

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(model, nn.parallel.DistributedDataParallel):
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.module.parameters(), create_graph=is_second_order)
        else:
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(logger, data_loader, model, device, adv=None, args=None, eval_header=None, indices_in_1k=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if indices_in_1k is not None:
                output = model(images)[:,indices_in_1k]
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_inc(logger, model, device, args=None, target_type=None):
    """Evaluate network on given corrupted dataset."""

    model.eval()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    result_dict = {}
    ce_alexnet = utils.get_ce_alexnet()

    # transform for imagenet-c
    inc_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    for name, path in utils.data_loaders_names.items():
        for severity in range(1, 6):
            if target_type is not None and (path != target_type or severity != 5):
                continue
            inc_dataset = torchvision.datasets.ImageFolder(os.path.join(args.inc_path, path, str(severity)), transform=inc_transform)

            sampler_val = torch.utils.data.DistributedSampler(
                inc_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            inc_data_loader = torch.utils.data.DataLoader(
                inc_dataset, sampler=sampler_val, batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            test_stats = evaluate(logger, inc_data_loader, model, device, args=args)
            logger.info(f"Accuracy on the {name+'({})'.format(severity)}: {test_stats['acc1']:.1f}%")
            result_dict[name+'({})'.format(severity)] = test_stats['acc1']
            if target_type is not None:
                return test_stats['acc1']

    mCE = 0
    counter = 0
    overall_acc = 0
    for name, path in utils.data_loaders_names.items():
        acc_top1 = 0
        for severity in range(1, 6):
            acc_top1 += result_dict[name+'({})'.format(severity)]
        acc_top1 /= 5
        CE = utils.get_mce_from_accuracy(acc_top1, ce_alexnet[name])
        mCE += CE
        overall_acc += acc_top1
        counter += 1
        logger.info("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
            name, acc_top1, 100. * CE))

    overall_acc /= counter
    mCE /= counter
    logger.info("Corruption Top1 accuracy {0:.2f}, mCE: {1:.2f}".format(overall_acc, mCE * 100.))
    return mCE * 100.


def AttentionDiversificationLoss(attn, th=2):
    sim_sum = 0
    counter = 1e-6
    for i in range(len(attn)):
        mask0 = attn[i].mean(dim=1).squeeze()
        n_tokens = mask0.shape[-1]
        threshold = th/n_tokens
        score0 = torch.mean(mask0, dim=1, keepdim=True)
        mask0 = (mask0 > threshold) * (mask0)
        score0 = (score0 > threshold) * (score0)
        sim = F.cosine_similarity(score0, mask0, dim=-1)
        sim = sim.mean()
        sim_sum += sim
        counter += 1
    sim_sum = sim_sum / counter
    return sim_sum
