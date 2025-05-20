import torch.nn as nn
from models.unet_precip_regression_lightning import UNetDSAttention

def load_encoder():
    hparams = {
        "n_channels": 12,
        "n_classes": 1,
        "bilinear": True,
        "reduction_ratio": 16,
        "kernels_per_layer": 2
    }
    base = UNetDSAttention(hparams=hparams)
    return nn.Sequential(
        base.inc,
        base.cbam1,
        base.down1,
        base.cbam2,
        base.down2,
        base.cbam3,
        base.down3,
        base.cbam4,
        base.down4,
        base.cbam5,
        nn.AdaptiveAvgPool2d((1, 1)))