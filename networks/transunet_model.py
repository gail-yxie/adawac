from networks.vit_seg_modeling import DecoderCup, VisionTransformer
import numpy as np


class DecoderCupLatent(DecoderCup):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size() # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        latents = [x]
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x, latents


class TransUnetLatent(VisionTransformer):
    def __init__(self, config, img_size, num_classes):
        super().__init__(
            config, img_size=img_size, num_classes=num_classes
        )
        self.decoder = DecoderCupLatent(config)
        print("Finish building TransUNetLatent")

    def forward(self, x, return_latent=False):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)  # (B, n_patch, hidden)
        x, latents = self.decoder(x, features)
        logits = self.segmentation_head(x)
        if return_latent:
            return logits, latents
        else:
            return logits
