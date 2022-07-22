import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# * means the paramter can be modify by CMD.
#------------------------------------------------------
#  The DATASET config
#------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.BATCH_SIZE = 64  #*
_C.DATASET.PATH = "data2/imagenet-1k" #*
_C.DATASET.NAME = "imagenet1k"  #*
_C.DATASET.INPUT_RESOLUTION = 224   #*
_C.DATASET.INTERPOLATION = "bicubic"
_C.DATASET.ZIP_MODE = False
_C.DATASET.CACHE_MODE = 'part'
_C.DATASET.WORKERS_NUMS = 8

#------------------------------------------------------
#  Model config
#------------------------------------------------------
#Base setting
_C.MODEL = CN()
_C.MODEL.TYPE = "DMS-Transformer"
_C.MODEL.NAME = "DMS_tiny_patch4_window7_k2_224"
_C.MODEL.NUMBER_CLASSES = 1000
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.ATTN_DROP_RATE = 0.0

#Model scale
_C.MODEL.DMS = CN()
_C.MODEL.DMS.PATCH_SIZE = 4
_C.MODEL.DMS.WINDOW_SIZE = 7
_C.MODEL.DMS.IN_CHANS = 3
_C.MODEL.DMS.EMBED_DIM = 96
_C.MODEL.DMS.DEPTHS = [2,2,6,2]
_C.MODEL.DMS.NUM_HEADS = [3,6,12,24]
_C.MODEL.DMS.MLP_RATE = 4
_C.MODEL.DMS.QKV_BIAS = True
_C.MODEL.DMS.QKV_SCALE = None
_C.MODEL.DMS.APE = False
_C.MODEL.DMS.DWPE = True
_C.MODEL.DMS.PATCH_NORM = True

#Dynamic settings
_C.MODEL.DMS.ATTENTION_TYPE = [['W','W'],["W","D"],["W","D","W","D","W","D"],["D","D"]]
_C.MODEL.DMS.DYNAMIC_FACTOR = [2, 4, 2, 1]
_C.MODEL.DMS.DYNAMIC_STRIDE = [2, 4, 2, 2]
_C.MODEL.DMS.CENTER_K = 2
_C.MODEL.DMS.POSTYPE = "rel"
_C.MODEL.DMS.CONV_PE = True
_C.MODEL.DMS.CONV_PM = True
_C.MODEL.RESUME = ""

#Training settings
_C.TRAIN = CN()
_C.TRAIN.EPOCH = 300
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.WARMUP_EPOCH = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.WARMUP_LR = 1e-6
_C.TRAIN.MIN_LR =1e-5
_C.TRAIN.ACCUMULATION_STEPS = 0    #*   Gradient accumulation
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.USE_CHECKPOINT = False      #*

#lr_scheduler settings
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 50
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

#optimizer settings
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


#Data augment settings
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.RAND_AUGMENT = "rand-m9-mstd0.5-inc1"   #"start with rand" means using rand-augment
_C.AUG.TRIAUG = True
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.RECOUNT = 1
_C.AUG.COLOR_JITTER = 0.4
#Erasing prob:
_C.AUG.ERASE_PROB = 0.25
#Erasing model:
#   'const' - erase block is constant color of 0 for all channels
#   'rand'  - erase block is same per-channel random (normal) color
#   'pixel' - erase block is per-pixel random (normal) color
_C.AUG.ERASE_MODEL = "pixel"
_C.AUG.ERASE_COUNT = 1



#Apex settings
_C.AMP_MODEL = "01"
_C.OUTPUT = ""
_C.SAVE_EPOCH = 1
_C.LOCAL_RANK = -1
_C.LOG_FREQ = 1


model_config_file = {
    "tiny": "configs/DMS_small_patch4_window7_k2_224.yaml",
    "small": "configs/DMS_small_patch4_window7_k2_224.yaml",
    "base": "configs/DMS_small_patch4_window7_k2_224.yaml",
}

def _update_config_from_file(config, model_type):
    assert model_type in ["small", "base", "tiny"] ,"Model variants not exist!"

    config.defrost()
    cfg_file = model_config_file[model_type]
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)      #将yaml文件中的配置移植到config中
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.model_type)
    config.defrost()
    # merge from specific arguments
    if args.batchsize:
        config.DATASET.BATCH_SIZE = args.batchsize
    if args.dataset:
        config.DATASET.DATASET = args.dataset
    if args.image_path:
        config.DATASET.PATH = args.image_path
    if args.zip:
        config.DATASET.ZIP_MODE = True
    if args.cache_mode:
        config.DATASET.CACHE_MODE = args.cache_mode
    if args.worker_num:
        config.DATASET.WORKERS_NUMS = args.worker_num
    if args.img_size:
        config.DATASET.INPUT_RESOLUTION = args.img_size

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.weight_decay:
        config.WEIGHT_DECAY = args.weight_decay
    if args.gradient_accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.gradient_accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_model:
        config.AMP_MODEL = args.amp_model
    if args.output_dir:
        config.OUTPUT = args.output_dir



    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config
