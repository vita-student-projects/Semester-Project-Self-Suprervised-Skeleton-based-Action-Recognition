# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import yaml
import numpy as np
import os
import os.path as osp
import time
from pathlib import Path
import sys

import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_temp_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS

from linprobe_engine import train_one_epoch, evaluate

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torchinfo import summary


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_args_parser():
    parser = argparse.ArgumentParser('Linear Probe Finetune action classification', add_help=False)
    parser.add_argument('--config', default='./config/ntu60_xsub_linear_parallel_GradNorm.yaml', help='path to the configuration file')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--debug', default=False, type=bool,
                        help='Debug the code or not')

    # Model parameters
    parser.add_argument('--model', default='ParallelFormer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')

    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--model_checkpoint_path', default='',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--feeder', default='feeder.feeder_ntu', help='data loader will be used')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--val_feeder_args', default=dict(), help='the arguments of data loader for validation')

    parser.add_argument('--output_dir', default='./output_dir_finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--nb_classes', default=60, type=int,
                        help='number of the classification types')
    parser.add_argument('--log_dir', default='./output_dir_finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def ddp_setup(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    args.distributed = True

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args)
    torch.cuda.set_device(rank)

    if rank==0: 
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset
    Feeder = import_class(args.feeder)
    dataset_train = Feeder(**args.train_feeder_args)
    dataset_val = Feeder(**args.val_feeder_args)
    if args.debug:
        subset_indices = list(range(int(len(dataset_train)/4)))
        dataset_train = torch.utils.data.Subset(dataset_train, subset_indices)
        dataset_val = torch.utils.data.Subset(dataset_val, subset_indices)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    def worker_init_fn(worker_id):                                                          
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if global_rank == 0:
        print('Train Dataset size: ', len(data_loader_train.dataset))
        print('Val Dataset size: ', len(data_loader_val.dataset))

    
    # define the model
    Model = import_class(args.model)
    model = Model(**args.model_args)

    if args.model_checkpoint_path and not args.eval:
        checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')
        if global_rank == 0:
            print("Load pre-trained checkpoint from: %s" % args.model_checkpoint_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.studentParallel.state_dict()

        checkpoint_student_parallel = {}
        for key, value in checkpoint_model.items():
            if 'studentParallel' in key:
                checkpoint_student_parallel[key] = value

        for k in ['head.fc.weight', 'head.fc.bias']:
            if k in checkpoint_student_parallel and checkpoint_student_parallel[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_student_parallel[k]

        # interpolate position embedding
        interpolate_temp_embed(model.studentParallel, checkpoint_student_parallel)

        msg = model.load_state_dict(checkpoint_student_parallel, strict=False)
        if global_rank == 0:
            print(msg)

        assert set(msg.missing_keys) == {'head.fc.weight', 'head.fc.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.fc.weight, std=0.01)

    # for linear prob only
    # hack: revise model's fc with BN
    model.head.fc = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.fc.in_features, affine=False, eps=1e-6), model.head.fc)
    
    # freeze all but the fc
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True


    model.to(device)
    model_without_ddp = model

    if rank == 0:
        summary_info = summary(model_without_ddp, [(8, 3, 100, 25, 2)])
        with open(osp.join(args.output_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(summary_info))
        with open(osp.join(args.output_dir, 'model_summary.txt'), 'a') as f:
            sys.stdout = f
            sys.stdout = sys.__stdout__

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if rank == 0:
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256
    if rank == 0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
    if rank == 0:
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[global_rank])
        model_without_ddp = model.module
    
    optimizer = torch.optim.SGD(model_without_ddp.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)

    loss_scaler = NativeScaler()

    
    criterion = torch.nn.CrossEntropyLoss()
    if rank == 0:
        print("criterion = %s" % str(criterion))

    if os.path.isfile(args.resume):
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        print("Start from scratch")

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        if rank == 0:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    if rank == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        misc.save_model_latest(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        if rank == 0:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        if rank == 0:
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
