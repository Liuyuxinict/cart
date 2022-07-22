from models.DMSTransformer import DLETransformer


def build(config):
    model_type = config.MODEL.TYPE
    print("loading the %s model" % model_type)
    model = DLETransformer(
        img_size=config.DATASET.INPUT_RESOLUTION,
        patch_size=config.MODEL.DMS.PATCH_SIZE,
        in_chans=config.MODEL.DMS.IN_CHANS,
        num_classes=config.MODEL.NUMBER_CLASSES,
        embed_dim=config.MODEL.DMS.EMBED_DIM,
        depths=config.MODEL.DMS.DEPTHS,
        num_heads=config.MODEL.DMS.NUM_HEADS,
        window_size=config.MODEL.DMS.WINDOW_SIZE,
        Attention_type=config.MODEL.DMS.ATTENTION_TYPE,
        dynamic_factor=config.MODEL.DMS.DYNAMIC_FACTOR,
        dynamic_stride=config.MODEL.DMS.DYNAMIC_STRIDE,
        center_k=config.MODEL.DMS.CENTER_K,
        postype=config.MODEL.DMS.POSTYPE,
        mlp_ratio=config.MODEL.DMS.MLP_RATE,
        qkv_bias=config.MODEL.DMS.QKV_BIAS,
        qk_scale=config.MODEL.DMS.QKV_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        conv_pe=config.MODEL.DMS.CONV_PE,
        conv_pm=config.MODEL.DMS.CONV_PM,
        ape=config.MODEL.DMS.APE,
        dwpe=config.MODEL.DMS.DWPE,
        patch_norm=config.MODEL.DMS.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT
    )
    return model
