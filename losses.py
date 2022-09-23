import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

# trim examples with zero labels
class TrimSparseLoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224):
        super().__init__()
        self.acc_loss_array = None
        self.img_size = img_size

    def forward(self, output, label, index):
        tmp_loss = F.cross_entropy(output, label, reduction="none")
        label_sum = torch.sum(label, dim=[1, 2]).detach()
        mask = label_sum.gt(0).to(torch.float32)
        total = self.img_size * self.img_size * sum(mask)
        loss = (
            1.0 / total * torch.sum(tmp_loss * mask[:, None, None])
            if total != 0
            else 0.0 * torch.sum(tmp_loss)
        )
        return loss, sum(mask)


class TrimLossWithRatio(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, ratio: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.ratio = 1 - 1280 / 2211 if ratio < 0 else ratio

    def forward(self, output, label, index):
        tmp_loss = F.cross_entropy(output, label, reduction="none")
        image_loss = tmp_loss.mean((1, 2)).detach()
        thre = torch.quantile(image_loss, self.ratio)
        mask = image_loss.gt(thre).to(torch.float32)
        loss = 1.0 / sum(mask) * torch.sum(image_loss * mask)
        return loss, sum(mask)


class CELoss(nn.Module):
    def __init__(self, num_examp: int, img_size: int = 224, lam: float = 0.2):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.acc_loss_array = None

    def forward(self, output, label, index):
        return self.ce_loss(output, label), label.shape[0]


class Losses:
    ce_loss = CELoss
    trim_sparse = TrimSparseLoss
    trim_ratio = TrimLossWithRatio


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.acc_loss_array = torch.zeros(1)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, index):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        # print(score.shape, y_sum.shape, loss.shape)
        return loss

    def forward(self, inputs, target, index, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], index)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
