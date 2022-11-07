import datetime
import os
import ml_collections
import argparse
import math
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
from datasets.dataset_aug import Synapse_training_pair, ACDC_training_pair


def get_basic_config(argin):
    config = ml_collections.ConfigDict(vars(argin))
    info = f"{config.dataset:s}_{config.model:s}_{config.img_size:d}_{config.seed:d}"
    dac_info = f"_en{str(config.dac_encoder):s}"
    loss_info = (
        f"_{config.loss:s}_trim-ratio{config.trim_ratio:.1f}_dice-ratio{config.dice_ratio:.1f}_lr-w{config.lr_w:.0e}"
    )
    vit_info = (
        f"_skip{config.n_skip:d}_vitpatch{config.patch_size:d}_{config.vit_name:s}"
        if config.model == "transunet"
        else ""
    )
    train_info = f"_epo{config.epochs:d}_bs{config.batch_size:d}_lr{config.lr:.0e}_lrs-{config.decay_lr:s}_mm{config.momentum:.2f}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%m-%d-%Y-%H-%M")
    config.exp_name = f"{info:s}_{dac_info:s}_{loss_info:s}_{vit_info:s}_{train_info:s}__{timestamp}"

    config.results_dir = (
        os.path.join("../results/", config.exp_name)
        if not config.results_dir
        else config.results_dir
    )
    config.resume = ""
    config.use_checkpoint = False
    return config


def get_synapse_config(config):
    list_root = "./lists/lists_Synapse"
    partial_dict = {
        "full": {
            "list_dir": list_root,
            "num_samples": 2211,
        },
        "half_vol": {
            "list_dir": f"{list_root}/half_vol",
            "num_samples": 1088,
        },
        "half_slice": {
            "list_dir": f"{list_root}/half_slice",
            "num_samples": 1111,
        },
        "half_vol_sparse": {
            "list_dir": f"{list_root}/half_vol_sparse",
            "num_samples": 1287,
        },
        "half_slice_sparse": {
            "list_dir": f"{list_root}/half_slice_sparse",
            "num_samples": 1118,
        },
        "trim": {
            "list_dir": f"{list_root}/trim",
            "num_samples": 1280,
        },
    }
    config.num_classes = 9
    config.base_dir = '../data/Synapse/train_npz'
    # for partial
    config.list_dir = partial_dict[config.partial]["list_dir"]
    config.num_samples = partial_dict[config.partial]["num_samples"]
    config.max_iterations = (
        math.ceil(config.num_samples / config.batch_size) * config.epochs
    ) if config.max_iterations == 30000 else config.max_iterations
    # for testing
    config.test_list_dir = "./lists/lists_Synapse"
    config.volume_path = "../data/Synapse/test_vol_h5"
    config.z_spacing = 1
    config.Dataset = Synapse_dataset
    config.TrainingPair = Synapse_training_pair
    return config


def get_acdc_config(config):
    config.num_classes = 4
    config.num_samples = 1324
    config.max_iterations = (
        math.ceil(config.num_samples / config.batch_size) * config.epochs
    ) if config.max_iterations == 30000 else config.max_iterations

    config.list_dir = "./lists/lists_ACDC"
    config.base_dir = "../data/ACDC"
    # for testing
    config.test_list_dir = "./lists/lists_ACDC"
    config.volume_path = "../data/ACDC"
    config.z_spacing = 5
    config.Dataset = ACDC_dataset
    config.TrainingPair = ACDC_training_pair
    return config


def get_b16_config(config):
    """Returns the ViT-B/16 configuration."""
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.activation = 'softmax'
    return config


def get_transunet_config(config):
    config.model = "transunet"
    config._MODEL = "transunet"
    config = get_b16_config(config)
    config.n_classes = config.num_classes

    # get_r50_b16_config
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = config.num_classes
    config.n_skip = 3
    config.activation = "softmax"

    if config.pretrained_path.find("R50") != -1:
        config.patches.grid = (
            int(config.img_size / config.patch_size),
            int(config.img_size / config.patch_size),
        )
        
    config.encoder_init_tag = 'init-R50+ViT-B_16'
    return config


def get_unet_config(config):
    config.model = "unet"
    config._MODEL = "unet"
    config.encoder_name = "resnet34"
    config.encoder_depth = 5
    config.encoder_weights = "imagenet"
    config.decoder_use_batchnorm = True
    config.decoder_channels = (256, 128, 64, 32, 16)
    config.in_channels = 1
    config.encoder_init_tag = 'init-' + config.encoder_name
    return config
