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
import timm.optim.optim_factory as optim_factory

import util.misc as misc
# from util.misc import NativeScalerWithGradNormCountBalanceLoss as NativeScaler
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from pretrain_engine_basic import train_one_epoch

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
    parser = argparse.ArgumentParser('CoupleFormer Pre-training', add_help=False)
    parser.add_argument('--config', default='./config/ntu60_xsub_pretrain_couple.yaml', help='path to the configuration file')

    parser.add_argument('--batch_size', default=6, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--debug', default=False, type=bool,
                        help='Debug the code or not')

    # Model parameters
    parser.add_argument('--model', default='SpatialFormer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')

    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed joints).')  
    parser.add_argument('--motion_stride', default=1, type=float,
                        help='')  
    parser.add_argument('--motion_aware_tau', default=0.80, type=float,
                        help='')  
                        
    parser.add_argument('--mask_ratio_inter', default=0.75, type=float,
                        help='Masking ratio inter (percentage of removed joints).')
    parser.add_argument('--mask_ratio_intra', default=0.80, type=float,
                        help='Masking ratio intra (percentage of removed joints).')

    parser.add_argument('--norm_skes_loss', default=True, type=bool,
                        help='')

    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--balanceLoss', type=bool, default=False,
                        help='')
    

    # Dataset parameters
    parser.add_argument('--feeder', default='feeder.feeder_ntu', help='data loader will be used')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=1, type=int)
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
    if args.debug:
        subset_indices = list(range(int(len(dataset_train)/400)))
        dataset_train = torch.utils.data.Subset(dataset_train, subset_indices)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

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

    if global_rank == 0:
        print('Train Dataset size: ', len(data_loader_train.dataset))
    
    # # define the model
    Model = import_class(args.model)
    model = Model(**args.model_args)

    model.to(device)

    model_without_ddp = model

    if rank == 0:
        summary_info = summary(model_without_ddp, [(8, 3, 100, 25, 2)])
        with open(osp.join(args.output_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(summary_info))
        with open(osp.join(args.output_dir, 'model_summary.txt'), 'a') as f:
            sys.stdout = f
            print(model_without_ddp)
            sys.stdout = sys.__stdout__

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if global_rank == 0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[global_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    # model._set_static_graph()
    param_groups = optim_factory.param_groups_layer_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if os.path.isfile(args.resume):
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        print("Start from scratch")

    if global_rank == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch+1) % 20 == 0 or epoch == 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        misc.save_model_latest(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if global_rank == 0:
        print('Training time {}'.format(total_time_str))

    destroy_process_group()
    

if __name__ == '__main__':
    parser = get_args_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()

        invalid_keys = []
        for k in default_args.keys():
            if k not in key:
                invalid_keys.append(k)
        for k in invalid_keys:
            del default_args[k]
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)