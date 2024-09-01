# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import util.misc as misc
import util.lr_sched as lr_sched
from torch.autograd import grad

import torch.nn.functional as F


def decoder_loss(target, pred, mask, norm_skes_loss):
    """
    imgs: [NM, TP*VP, t_patch_size * patch_size * C]
    pred: [NM, TP*VP, t_patch_size * patch_size * C]
    mask: [NM, TP, VP], 0 is keep, 1 is remove,
    """
    if norm_skes_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch
    B, T, J = mask.shape
    mask = mask.reshape(B, T*J)
    
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed joints

    return loss
    
def motion_loss(studentMotionPredict, teacherMotionGT, mask, norm_skes_loss):
    """
    imgs: [NM, TP*VP, t_patch_size * patch_size * C]
    pred: [NM, TP*VP, t_patch_size * patch_size * C]
    mask: [NM, TP, VP], 0 is keep, 1 is remove,
    """
    teacherMotionGT = teacherMotionGT.detach()
    if norm_skes_loss:
        # mean = studentMotionPredict.mean(dim=-1, keepdim=True)
        # var = studentMotionPredict.var(dim=-1, keepdim=True)
        # studentMotionPredict = (studentMotionPredict - mean) / (var + 1.0e-6) ** 0.5

        mean = teacherMotionGT.mean(dim=-1, keepdim=True)
        var = teacherMotionGT.var(dim=-1, keepdim=True)
        teacherMotionGT = (teacherMotionGT - mean) / (var + 1.0e-6) ** 0.5

    loss = (studentMotionPredict - teacherMotionGT) ** 2
    loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch
    # print(loss.shape)
    loss = loss.sum() / mask.sum()  # mean loss on removed joints
    return loss.sum()

# def motion_entropy_loss(studentMotionPredict, teacherMotionGT, center, norm_skes_loss):
#     """
#     imgs: [NM, TP*VP, t_patch_size * patch_size * C]
#     pred: [NM, TP*VP, t_patch_size * patch_size * C]
#     mask: [NM, TP, VP], 0 is keep, 1 is remove,
#     """
#     teacherMotionGT = teacherMotionGT.detach()
#     teacher_probs = F.softmax((teacher_output - center) / tau_t, dim=2)
#     student_probs = F.log_softmax(student_output / tau_s, dim=2)

#     loss = - (teacher_probs * student_probs).sum(dim=2).mean()

#     return loss

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None and misc.is_main_process():
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.float().to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            studentProject, target, studentMotionPredict, teacherMotionGT, mask = model(samples,
                                            mask_ratio=args.mask_ratio,
                                            motion_stride=args.motion_stride,
                                            motion_aware_tau=args.motion_aware_tau)

            loss_stu_tea = motion_loss(studentMotionPredict, teacherMotionGT, mask, args.norm_skes_loss)
            loss_decoder = decoder_loss(target, studentProject, mask, args.norm_skes_loss)
            # loss_control = torch.sigmoid(model.module.adjust_para)
            # loss_control = torch.clamp(loss_control, max=0.8)
            # loss = loss_stu_tea*loss_control + loss_decoder*(1-loss_control)
            loss = loss_stu_tea * 10 + loss_decoder

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(11)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.module._ema_update_teacher_encoder()
        metric_logger.update(loss=loss_value)


        # if not args.balanceLoss:
        #     loss /= accum_iter
        #     loss_scaler(loss, optimizer, parameters=model.parameters(),
        #                 update_grad=(data_iter_step + 1) % accum_iter == 0)
        #     with torch.no_grad():
        #         model.module._ema_update_teacher_encoder()
        #     if (data_iter_step + 1) % accum_iter == 0:
        #         optimizer.zero_grad()

        #     torch.cuda.synchronize()
        #     metric_logger.update(loss=loss_value)
        #     metric_logger.update(loss_control=loss_control)
        # else:
        #     _, loss_balanced = loss_scaler(torch.stack([loss_decoder, loss_stu_tea]), optimizer, 
        #                 parameters=model.parameters(), 
        #                 shared_layers=model.module.studentParallel.norm,
        #                 update_grad=(data_iter_step + 1) % accum_iter == 0)
        #     with torch.no_grad():
        #         model.module._ema_update_teacher_encoder()
        #     if (data_iter_step + 1) % accum_iter == 0:
        #         optimizer.zero_grad()

        #     torch.cuda.synchronize()
        #     metric_logger.update(loss_balanced=loss_balanced)

        metric_logger.update(loss_decoder=loss_decoder)
        metric_logger.update(loss_motion=loss_stu_tea)
            

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}