""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, config, img_size, num_classes):
        super(UNet, self).__init__()
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        # self.bilinear = bilinear
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        # change inchannels to 3
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def load_from(self, config):
        pass


class UNetLatent(UNet):
    def __init__(self, config, img_size, num_classes):
        # n_classes = config.n_classes
        # bilinear = config.bilinear
        super().__init__(config, img_size, num_classes)
        print("Finish building UNetLatent")

    def forward(self, x, return_latent=False):
        # change inchannels to 3
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        latents = [x5]
        if return_latent:
            return logits, latents
        else:
            return logits