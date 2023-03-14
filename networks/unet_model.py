# Code adapted from: https://github.com/qubvel/segmentation_models.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UNetLatent(smp.Unet):
    def __init__(self, config, img_size, num_classes):
        super(UNetLatent, self).__init__(
            encoder_name=config.encoder_name,
            encoder_depth=config.encoder_depth,
            encoder_weights=config.encoder_weights,
            decoder_use_batchnorm=config.decoder_use_batchnorm,
            decoder_channels=config.decoder_channels,
            decoder_attention_type=None,
            in_channels=config.in_channels,
            classes=config.num_classes,
            activation=None,
            aux_params=None,
        )
    
    def forward(self, x, return_latent=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        logits = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return logits, labels
        if return_latent:
            return logits, features[-1:]
        else:
            return logits
    
    def load_from(self, config):
        pass


class UNet(UNetLatent):
    def __init__(self, config, img_size, num_classes):
        super(UNet, self).__init__(config, img_size, num_classes)

    def forward(self, x):
        logits = super(UNet, self).forward(x)
        return logits
