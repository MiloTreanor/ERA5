import torch
import torch.nn as nn
from torchvision import transforms


class ApplyMeanFilter:
    """Vectorized mean filter for all 12 channels"""

    def __init__(self, kernel_size=5):
        self.conv = nn.Conv2d(12, 12, kernel_size,
                              padding=kernel_size // 2,
                              groups=12,
                              bias=False)
        nn.init.constant_(self.conv.weight, 1 / (kernel_size ** 2))
        self.conv.requires_grad_(False)

    def __call__(self, img):
        return self.conv(img.unsqueeze(0)).squeeze(0)


class TemporalAugmentation:
    def __init__(self):
        self.crop = transforms.RandomResizedCrop(
            288, scale=(0.9, 0.9),
            ratio=(1., 1.),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.mean_filter = ApplyMeanFilter(kernel_size=5)

    def __call__(self, x):

        if torch.rand(1) < 0.7:
            x = self.crop(x)

        if torch.rand(1) < 0.4:
            x = self.mean_filter(x)

        return x