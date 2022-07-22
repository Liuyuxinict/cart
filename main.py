import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp

from models.DMSTransformer import DLETransformer
from models.build_models import build

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config import get_config

def parse_option():
    parser = argparse.ArgumentParser('DMS Transformer for training and testing')
    parser.add_argument('--model_type', type=str, default="small", help="base, small, tiny")

    parser.add_argument('--batchsize', default=16, type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='imagenet1k', help='imagenet, food101, food172')
    parser.add_argument('--image_path', type=str, help='path to dataset', default="")             #Must be input by line command
    parser.add_argument('--zip', action='store_true', default=False,
                        help='use zipped dataset instead of folder dataset')
    # 这里后面再看
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp_model', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--output_dir', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--worker_num', help='the num of workers during train and test',
                        default = 8, type=int)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def settings(args, config):
    #setting distribution

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    #########################################  待会再写
    torch.cuda.set_device(config.LOCAL_RANK)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    #torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    #torch.distributed.barrier()

    model = build(config)
    print(model)


def main():
    args, config = parse_option()
    print(config)
    settings(args, config)


if __name__ == "__main__":
    main()