import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch.nn as nn
import wandb

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    # elif pred.sum() > 0 and gt.sum() == 0:
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def calculate_metric_percase_previous(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=[256, 256],
    test_save_path=None,
    case='',
    z_spacing=1,
    metric_choice="modified",
    wandb_save=False,
):
    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(
                    slice, (patch_size[0] / x, patch_size[1] / y), order=3
                )  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    metric_list = []
    for i in range(1, classes):
        if metric_choice == "modified":
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
        elif metric_choice == "previous":
            metric_list.append(
                calculate_metric_percase_previous(prediction == i, label == i)
            )
        else:
            raise ValueError("Not valid test metric.")

    if wandb_save:
        class_labels = {
            0:'background', 
            1:'aorta', 
            2:'gallbladder', 
            3:'left kidney', 
            4:'right kidney', 
            5:'liver', 
            6:'pancreas', 
            7:'spleen', 
            8:'stomach',
        }
        for i in range(0, prediction.shape[0], max(prediction.shape[0]//30, 1)):
            mask_img = wandb.Image(image[i], masks={
                "predictions": {
                    "mask_data": prediction[i].astype(int),
                    "class_labels": class_labels
                },
                "labels": {
                    "mask_data": label[i],
                    "class_labels": class_labels
                }
            })
            wandb.log({f'test_{case}': mask_img})

    if test_save_path:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + "/" + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + "/" + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + "/" + case + "_gt.nii.gz")
    return metric_list


class DiceLossFull(nn.Module):
    def __init__(self, config):
        super(DiceLossFull, self).__init__()
        self.n_classes = config.num_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5  # / target.size(dim=0)
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1.0 - loss
        return loss.repeat(target.size(dim=0))  # (B,)

    def forward(self, inputs, target, weight=None, softmax=False):
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
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes  # (B,)