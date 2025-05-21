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
        base.down1,
        base.down2,
        base.down3,
        base.down4)