# Unet model wrapper that additionally returns features from the intermediate layers for consistency regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DiceLossFull

# TransUNet
from networks.transunet_model import TransUnetLatent
from networks.vit_seg_modeling import VisionTransformer as TransUNet

from networks.unet_model import UnetLatent, UNet


class UNETS_AW_AC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLossFull(config)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.dac_loss = nn.MSELoss(reduction="none")
        self.dac_coef = [config.dac_encoder]

        PAIR_MODEL = {
            "transunet": TransUnetLatent,
            'unet': UnetLatent,
        }
        self.model = PAIR_MODEL[config._MODEL](
            config, config.img_size, config.num_classes
        )

        self.weights = (torch.ones(config.num_samples, 2) / 2.0).cuda()
        self.ratio = (
            config.trim_ratio if config.trim_ratio >= 0 else 1 - 1280 / 2211
        )  # default trim ratio for Synapse

    def forward(self, x1, y1, idx1, x2, y2, idx2):
        """
        samples.keys() = {'image', 'label', 'case_name', 'idx'}
        """
        logits1, latents1 = self.model(x1, return_latent=True)
        logits2, latents2 = self.model(x2, return_latent=True)

        # class
        loss_ce1 = torch.mean(self.ce_loss(logits1, y1.long()), dim=[1, 2])
        loss_ce2 = torch.mean(self.ce_loss(logits2, y2.long()), dim=[1, 2])
        loss_ce = (loss_ce1 + loss_ce2) * 0.5  # (B,)

        # dice
        loss_dice1 = self.dice_loss(logits1, y1, softmax=True)
        loss_dice2 = self.dice_loss(logits2, y2, softmax=True)
        loss_dice = (loss_dice1 + loss_dice2) * 0.5  # (B,)

        # srw-dac
        dac_reg = torch.zeros_like(loss_ce)
        weights = self.weights[idx1].clone().detach()  # (B,2)

        loss_ce_ = loss_ce.clone().detach()
        dac_reg_ = dac_reg.clone().detach()  # if not dac, it's zeros
        batch_num = x1.shape[0]
        mask = None

        if self.config.dac:
            # dac only
            for i in range(len(self.dac_coef)):
                dac_reg += (
                    torch.mean(self.dac_loss(latents1[i], latents2[i]), dim=[1, 2, 3])
                    * self.dac_coef[i]
                )
            dac_reg_ = dac_reg.clone().detach()
            # mask = torch.ones(batch_num).cuda()

            # different losses with dac
            if self.config.loss == "adawac":
                weights[:, 0] *= torch.exp(self.config.lr_w * loss_ce_)
                weights[:, 1] *= torch.exp(self.config.lr_w * dac_reg_)
                weights = F.normalize(weights, p=1, dim=1)
                self.weights[idx1] = weights
            elif self.config.loss in ["trim-train",'pseudo']:
                label_sum = torch.sum(y1, dim=[1, 2]).clone().detach()
                mask = label_sum.gt(0).to(torch.float32)
            elif self.config.loss == "trim-ratio":
                thre = torch.quantile(loss_ce_, self.ratio)
                mask = loss_ce_.gt(thre).to(torch.float32)

            if self.config.loss=='pseudo':
                assert mask is not None
                loss = self.config.dice_ratio * loss_dice + (1.0 - self.config.dice_ratio) * (loss_ce * mask + dac_reg * (1.0-mask))
            elif "trim" in self.config.loss:  # without mask version
                assert mask is not None
                batch_num = sum(mask).item()  # type: ignore
                loss = (
                    self.config.dice_ratio * loss_dice
                    + (1.0 - self.config.dice_ratio) * (loss_ce + dac_reg) / 2.0
                ) * mask
            else: # adawac
                loss = (
                    self.config.dice_ratio * loss_dice
                    + (1.0 - self.config.dice_ratio) * (loss_ce * weights[:, 0]
                    + dac_reg * weights[:, 1])
                )

        elif self.config.loss == "reweight-only":
            loss_ce1_ = loss_ce1.clone().detach()
            weights[:, 0] *= torch.exp(self.config.lr_w * loss_ce1_)
            weights = F.normalize(weights, p=1, dim=1)
            self.weights[idx1, 0] = weights[:, 0]
            loss = (
                self.config.dice_ratio * loss_dice1
                + (1.0 - self.config.dice_ratio) * loss_ce1 * weights[:, 0]
            )
        elif self.config.loss == "base":
            loss = (
                self.config.dice_ratio * loss_dice1
                + (1.0 - self.config.dice_ratio) * loss_ce1
            )
        elif self.config.loss == "pair":
            loss = (
                self.config.dice_ratio * loss_dice
                + (1.0 - self.config.dice_ratio) * loss_ce
            )
        else:
            raise ValueError("Unknown loss type: {}".format(self.config.loss))

        if "trim" in self.config.loss:
            loss = (
                torch.sum(loss) / batch_num if batch_num != 0 else 0.0 * torch.sum(loss)
            )
        else:
            loss = torch.mean(loss)
        loss_dice_ = loss_dice.clone().detach()
        weights_ = weights.clone().detach()
        logits1_ = logits1.clone().detach()
        return loss, loss_ce_, loss_dice_, dac_reg_, weights_, logits1_, batch_num

    def load_from(self, config):
        self.model.load_from(config)


# model with plain dataloader to load single data instead of pair data
class UNETS_BASE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLossFull(config)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        PAIR_MODEL = {
            "transunet": TransUNet,
            'unet': UNet,
        }
        self.model = PAIR_MODEL[config._MODEL](
            config, config.img_size, config.num_classes
        )

        # define losses
        self.ratio = (
            config.trim_ratio if config.trim_ratio >= 0 else 1 - 1280 / 2211
        )  # default trim ratio for Synapse

    def forward(self, x, y, idx, x2, y2, idx2):
        # We can use this for simplicity, but here we use the original model to reproduce the results
        # logits = self.model(x, return_latent=False)
        logits = self.model(x)
        loss_ce = torch.mean(self.ce_loss(logits, y.long()), dim=[1, 2])
        loss_ce_ = loss_ce.clone().detach()

        loss_dice = self.dice_loss(logits, y, softmax=True)
        loss_dice_ = loss_dice.clone().detach()
        batch_num = x.shape[0]

        # just for recording consistency
        logits_ = logits.clone().detach()
        weights_ = self.weights[idx].clone().detach()
        dac_reg_ = torch.zeros_like(loss_ce)

        if "trim" in self.config.loss:
            mask = None
            if self.config.loss == "trim-train":
                label_sum = torch.sum(y, dim=[1, 2]).clone().detach()
                mask = label_sum.gt(0).to(torch.float32)
            elif self.config.loss == "trim-ratio":
                thre = torch.quantile(loss_ce_, self.ratio)
                mask = loss_ce_.gt(thre).to(torch.float32)
            else:
                raise ValueError("Unknown loss type: {}".format(self.config.loss))

            assert mask is not None
            batch_num = sum(mask).item()  # type: ignore
            loss = (
                self.config.dice_ratio * loss_dice
                + (1.0 - self.config.dice_ratio) * loss_ce
            ) * mask
            loss = (
                torch.sum(loss) / batch_num if batch_num != 0 else 0.0 * torch.sum(loss)
            )
        else:
            loss = (
                self.config.dice_ratio * loss_dice
                + (1.0 - self.config.dice_ratio) * loss_ce
            )
            loss = torch.mean(loss)

        return loss, loss_ce_, loss_dice_, dac_reg_, weights_, logits_, batch_num

    def load_from(self, config):
        self.model.load_from(config)
