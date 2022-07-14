import itertools
import os
import random
from typing import Any
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, list_dir=None, split="train", transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx].strip("\n") + ".h5"
        if self.split == "train":
            h5f = h5py.File(
                self._base_dir + "/ACDC_training_slices/{}".format(case), "r"
            )
            h5f: Any
            image = h5f["image"][:]
            label = h5f["label"][:]  # fix sup_type to label
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            h5f = h5py.File(
                self._base_dir + "/ACDC_training_volumes/{}".format(case), "r"
            )
            image = h5f["image"][:]
            label = h5f["label"][:]
            sample = {"image": image, "label": label}
        sample["idx"] = idx
        sample["case_name"] = case.replace(".h5", "")
        return sample, None


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )  # the default is 0
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )

        assert (image.shape[0] == self.output_size[0]) and (
            image.shape[1] == self.output_size[1]
        )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
