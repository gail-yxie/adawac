# Code adapted from: https://github.com/milesial/Pytorch-UNet
from networks.unet_parts import DoubleConv, Down, Up, OutConv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UnetLatent(nn.Module):
    def __init__(self, config, img_size, num_classes, bilinear=False):
        super(UnetLatent, self).__init__()
        self.config = config
        self.n_channels = 1
        self.n_classes = config.num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x, return_latent=False):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # return (with latents)
        if return_latent:
            return logits, [x5]
        else:
            return logits
        
    def load_from(self, config):
        pass


class UNet(UnetLatent):
    def __init__(self, config, img_size, num_classes, bilinear=False):
        super(UNet, self).__init__(config, img_size, num_classes, bilinear=bilinear)

    def forward(self, x):
        logits = super(UNet, self).forward(x)
        return logits
