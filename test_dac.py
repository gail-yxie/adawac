import argparse
from typing import Any
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from configs import get_basic_config, get_transunet_config, get_synapse_config, get_acdc_config
from models import UNETS_AW_AC, UNETS_BASE
from train_dac import inference
import os

import wandb

# test
def test():
    parser = argparse.ArgumentParser(description='Volumetric Medical Image Segmentation')
    # dataset
    parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
    parser.add_argument('--partial', type=str, default="full", 
                        choices=["full", "half_vol", "half_slice", "half_vol_sparse", "half_slice_sparse", "trim"], help="choose partial samples of a full list")
    parser.add_argument('--batch-size', type=int, default=24, help='batch_size per gpu')

    # hardware and randomness
    parser.add_argument('--num-gpu', default=1, type=int, help='number of GPU. Check with nvidia-smi')
    parser.add_argument('--num-workers', default=8, type=int, help='num_worksd for dataloader')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    
    # training and optimization
    parser.add_argument('--max-iterations', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='baseline initial learning rate')
    parser.add_argument('--decay=lr', type=str, default='no_decay', help='type of learning rate schedule')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    
    # model architecture
    parser.add_argument('--model', type=str, default='transunet', help='model network')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--n-skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit-name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit-patches-size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--img-size', default=224, type=int, help='Input image size to model (independent of raw image size)')
    parser.add_argument('--data-choice', type=str, default='pair', choices=['pair', 'single'], help='choose pair or single')

    # loss
    parser.add_argument('--loss', type=str, default='base', 
                        choices=['base', 'pair', 'adawac', 'trim_sparse', 'trim-train', 'reweight-only', 'dac-only'], help='loss function')
    parser.add_argument('--trim-ratio', type=float, default=0.0, help='trim ratio for trim-ratio loss')
    parser.add_argument('--dice-ratio', type=float, default=0.5, help='ratio for dice loss')
    
    # adaptive weighted agumentation consistency
    parser.add_argument('--dac', action='store_true', help='use DAC')
    parser.add_argument('--dac-encoder', default=1.0, type=float, help='regularization hyper-parameter for DAC')
    parser.add_argument('--lr-w', default=1.0, type=float, help='learning rate for weights')
    
    # experiment and results
    parser.add_argument('--exp-mark', type=str, default='', help='deprecated')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument("--test_save_path", type=str, default='',help="test dir to save predictions")

    arch_config = {'transunet': get_transunet_config}
    data_config = {'Synapse': get_synapse_config, 'ACDC': get_acdc_config}
    config = parser.parse_args()
    config = data_config[config.dataset](config)
    config = arch_config[config.model](config)
    config = get_basic_config(config)

    config: Any

    if not config.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # load model
    model = UNETS_AW_AC(config=config) if config.data_choice == 'pair' else UNETS_BASE(config=config)
    model_name = 'model_last.pth' if config.dataset == 'Synapse' else 'best_model.pth'
    save_model_path = os.path.join(config.results_dir, model_name)
    if config.dataset == 'Synapse':
        checkpoint = torch.load(save_model_path)['state_dict']
        model.load_state_dict(checkpoint)
    if config.dataset == 'ACDC':
        checkpoint = torch.load(save_model_path)
        model.model.load_state_dict(checkpoint)

    # model parallel
    if config.num_gpu > 1:
        model = nn.DataParallel(model)
    else:
        model = model.cuda()

    # training logging
    wandb.init(project=f"transunet_{config.dataset}_simple", entity="medical-image", tags=[config.partial, config.model])
    wandb.config.update(config)
    inference(model.model, config, "modified")
    wandb.finish()
    return "Inference Finished!"
        

if __name__=='__main__':
    test()
    