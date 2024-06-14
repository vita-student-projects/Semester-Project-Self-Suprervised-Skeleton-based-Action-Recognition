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

USE_GPU = True
if USE_GPU and torch.cuda.is_available():

    def to_device(x, gpu=None):
        x = x.cuda(gpu)
        return x

else:

    def to_device(x, gpu=None):
        return x


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

    # for data_iter_step, (samples, label, index) in enumerate(data_loader):
    for data_iter_step, (samples, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.float().to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            studentencoded, teacherencoded, studentpredict, teachermasked, mask = model(samples)

            loss = (studentpredict - teachermasked) ** 2
            loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch
            # print(loss.sum(), mask.sum())
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed joints
            # print(loss)

            # train_loss = to_device(nn.MSELoss())
            # loss = train_loss(studentpredict, teachermasked)
        # for param_tea in model.module.teacherspatial.parameters():
        #     print(param_tea.requires_grad)  # not update by gradient
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(11)

        loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        optimizer.zero_grad()
        loss.backward()
        # for name, param in model.module.studentspatial.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name} has gradient computed.")
        # for name, param in model.module.teacherspatial.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name} has gradient computed.")
        optimizer.step()
        with torch.no_grad():
            model.module._ema_update_teacher_encoder()

        def check_trainable_parameters(model):
            trainable_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(name)
            return trainable_params

        # trainable_parameters = check_trainable_parameters(model.module.teacherspatial)
        # print("Trainable Parameters: ", trainable_parameters)
        # trainable_parameters = check_trainable_parameters(model.module.studentspatial)
        # print("Trainable Parameters: ", trainable_parameters)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

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