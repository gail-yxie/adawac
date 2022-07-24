# code in this file is adpated from:
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
# https://github.com/Beckschen/TransUNet/blob/main/datasets/dataset_synapse.py
import os
from datasets.dataset_synapse import RandomGenerator
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import h5py
import torch


# set random seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_loader(config):
    if config.data_choice == "single":
        transform = transforms.Compose(
            [RandomGenerator([config.img_size, config.img_size])]
        )
        dataset = config.Dataset(config.base_dir, config.list_dir, "train", transform)
    else:
        dataset = config.TrainingPair(config)
    print("Train set length = {:d}".format(len(dataset)))

    #### previous dataloader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     worker_init_fn=lambda id: random.seed(config.seed + id),
    # )

    #### add new worker function with generator
    g = torch.Generator()
    g.manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return dataloader


def get_test_loader(config):
    base_dir, list_dir = config.volume_path, config.test_list_dir
    split = "test_vol"
    dataset = config.Dataset(base_dir, list_dir, split)
    print("Test set length = {:d}".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return dataloader, dataset


# Two-view dataset, supervised
class Synapse_training_pair(Dataset):
    def __init__(self, config):
        self.config = config
        self.base_dir = config.base_dir
        self.list_dir = config.list_dir
        self.split = "train"
        self.sample_list = open(
            os.path.join(self.list_dir, self.split + ".txt")
        ).readlines()
        self.transform = transforms.Compose(
            [RandomGenerator([config.img_size, config.img_size])]
        )

    def __len__(self):
        return len(self.sample_list)

    def _get_transformed_sample(self, idx):
        slice_name = self.sample_list[idx].strip("\n")
        data_path = os.path.join(self.base_dir, slice_name + ".npz")
        data = np.load(data_path)
        image, label = data["image"], data["label"]
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = slice_name  # type: ignore
        sample["idx"] = idx
        return sample

    def __getitem__(self, idx):
        sample_1 = self._get_transformed_sample(idx)
        sample_2 = self._get_transformed_sample(idx)
        return sample_1, sample_2


class ACDC_training_pair(Synapse_training_pair):
    def _get_transformed_sample(self, idx):
        slice_name = self.sample_list[idx].strip("\n")
        data_path = (
            self.base_dir + "/ACDC_training_slices/{}".format(slice_name) + ".h5"
        )
        h5f = h5py.File(data_path, "r")
        image, label = h5f["image"][:], h5f["label"][:]  # type: ignore
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = slice_name
        sample["idx"] = idx
        return sample
