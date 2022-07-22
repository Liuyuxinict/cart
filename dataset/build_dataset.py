import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from dataset.samplers import SubsetRandomSampler

def build_loader(config):
    train_dataset = create_dataset(istrain=True, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    val_dataset = create_dataset(istrain=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=config.DATASET.BATCH_SIZE,
        num_workers=config.DATASET.WORKERS_NUMS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATASET.WORKERS_NUMS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=False
    )

    mixup = Mixup(
        mixup_alpha=config.AUG.MIXUP_ALPHA,
        cutmix_alpha=config.AUG.CUTMIX_ALPHA,
        cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.PROB,
        switch_prob=config.AUG.SWITCH_PROB,
        mode=config.AUG.MODEL,
        label_smoothing=config.LABEL_SMOOTH,
        num_classes=config.MODEL.NUMBER_CLASSES
    )

    return train_dataset, val_dataset, data_loader_train, data_loader_val, mixup




def build_transform(istrain, config):
    transform_total = []
    if istrain:
        transform = create_transform(
            input_size=config.DATASET.INPUT_RESOLUTION,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER,
            re_mode=config.AUG.ERASE_MODEL,
            re_prob=config.AUG.ERASE_PROB,
            re_count=config.AUG.ERASE_COUNT,
            auto_augment=config.AUG.RAND_AUGMENT,
            interpolation=config.DATASET.INTERPOLATION
        )
        return transform

    if config.TEST.CENTER_CROP:
        size = int((256.0/224.0) * config.DATASET.INPUT_RESOLUTION)
        transform_total.append(
            transforms.Resize(size,interpolation=_pil_interp(config.DATA.INTERPOLATION))
        )
    else:
        transform_total.append(
            transforms.Resize(config.DATASET.INPUT_RESOLUTION, interpolation=_pil_interp(config.DATA.INTERPOLATION))
        )
    transform_total.append(transforms.ToTensor())
    transform_total.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(transform_total)
    return transform


def create_dataset(istrain, config, eval):
    transform = build_transform(istrain, config)
    if config.DATASET.NAME == "imagenet1k":
        if eval:
            #waiting for completing.
            return
        if istrain:
            root = os.path.join(config.DATASET.PATH, "train")
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            root = os.path.join(config.DATASET.PATH, "val")
            dataset = datasets.ImageFolder(root, transform=transform)
    config.defrost()
    config.MODEL.NUMBER_CLASSES = 1000
    config.freeze()

    if config.DATASET.NAME == "food101":
        pass
    if config.DATASET.NAME == "food172":
        pass

    return dataset