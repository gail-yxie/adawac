import argparse
import math
from typing import Any
import os, sys
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from configs import get_basic_config, get_transunet_config, get_unet_config, get_synapse_config, get_acdc_config
from datasets.dataset_aug import get_train_loader, get_test_loader
from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
from models import UNETS_AW_AC, UNETS_BASE
from utils import test_single_volume

import wandb

# train
def train():
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
    parser.add_argument('--decay-lr', type=str, default='no_decay', help='type of learning rate schedule')
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
                        choices=['ce', 'base', 'pair', 'adawac', 'trim-ratio', 'trim-train', 'reweight-only', 'dac-only'], help='loss function')
    parser.add_argument('--trim-ratio', type=float, default=0.0, help='trim ratio for trim-ratio loss')
    parser.add_argument('--dice-ratio', type=float, default=0.5, help='ratio for dice loss')
    
    # adaptive weighted agumentation consistency
    parser.add_argument('--dac', action='store_true', help='use DAC')
    parser.add_argument('--dac-encoder', default=1.0, type=float, help='regularization hyper-parameter for DAC')
    parser.add_argument('--dac-decoder', default=[0.0, 0.0, 0.0, 0.0], nargs='+', type=float, 
                        help='hyper-parameters on decoder layers for sample-wisely reweighting DAC regularization')
    parser.add_argument('--lr-w', default=1.0, type=float, help='learning rate for weights')
    
    # experiment and results
    parser.add_argument('--exp-mark', type=str, default='', help='exp-mark')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument("--test_save_path", type=str, default='',help="test dir to save predictions")


    arch_config = {'transunet': get_transunet_config, 'unet': get_unet_config}
    data_config = {'Synapse': get_synapse_config, 'ACDC': get_acdc_config}
    config = parser.parse_args()
    config = data_config[config.dataset](config)
    config = arch_config[config.model](config)
    config = get_basic_config(config)

    config: Any

    # set randomness
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
    
    # dataloader
    trainloader = get_train_loader(config)
    config.max_iterations = config.epochs * len(trainloader) 
    ## initial for validation
    if config.dataset == 'ACDC':
        db_val = ACDC_dataset(base_dir=config.base_dir, list_dir=config.list_dir, split="val")
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    else:
        db_val, valloader = None, None

    # model
    model = UNETS_AW_AC(config=config) if config.data_choice == 'pair' else UNETS_BASE(config=config)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    
    # load pretrained / resume model
    epoch_start = 1
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(config.resume))
    else:
        model.load_from(config=config)

    # model parallel
    if config.num_gpu > 1:
        model = nn.DataParallel(model)
    else:
        model = model.cuda()

    # training logging
    if not os.path.exists(config.results_dir): os.makedirs(config.results_dir)

    logging.basicConfig(filename=os.path.join(config.results_dir, f'log.txt'), 
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    
    num_logger = {  "ce":   torch.zeros(config.num_samples, config.epochs).cuda(),
                    "dice": torch.zeros(config.num_samples, config.epochs).cuda(), 
                    "dac":  torch.zeros(config.num_samples, config.epochs).cuda(),
                    "w_dac":torch.zeros(config.num_samples, config.epochs).cuda(),
                }
    wandb.init(project=f"transunet_{config.dataset}_simple", entity="medical-image", tags=[config.partial, config.model])
    wandb.config.update(config)
    iter_num = 0
    # record
    best_modified, best_previous, best_performance, best_epoch = [0, 0], [0, 0], 0.0, epoch_start
    modified_dice, modified_hd, previous_dice, previous_hd = None, None, None, None

    # training loop
    for epoch in range(epoch_start, config.epochs+1):
        iter_num = train_epoch(model, trainloader, optimizer, epoch, config, num_logger, iter_num)
        # save model
        save_interval, valid_interval = 150, 10

        if epoch % save_interval == 0 and config.dataset == 'Synapse':  # type: ignore
            save_model_path = os.path.join(config.results_dir, 'model_last.pth')  # type: ignore
            logging.info("save model to {}".format(save_model_path))
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, save_model_path)
            # test
            modified_dice, modified_hd = inference(model.model, config, metric_choice="modified")
            previous_dice, previous_hd = inference(model.model, config, metric_choice="previous")
            if modified_dice > best_modified[0]:
                best_modified, best_epoch = [modified_dice, modified_hd], epoch
            if previous_dice > best_previous[0]:
                best_previous = [previous_dice, previous_hd]
        elif  epoch % valid_interval == 0 and config.dataset == "ACDC":
            assert valloader is not None
            assert db_val is not None
            best_performance, best_epoch = valid_epoch(model.model, config, valloader, db_val, best_performance, epoch, best_epoch)
    
    # test
    if config.dataset == "ACDC" and epoch > valid_interval:  # type: ignore
        save_best = os.path.join(config.results_dir, "best_model.pth")
        model.model.load_state_dict(torch.load(save_best))  # type: ignore
        logging.info("Loaded best model via validation.")
        best_modified = inference(model.model, config, metric_choice="modified")
        best_previous = inference(model.model, config, metric_choice="previous")

    wandb.log({ "best modified test mean dice": best_modified[0],
                "best modified test mean hd95": best_modified[1],
                "best previous test mean dice": best_previous[0],
                "best previous test mean hd95": best_previous[1],
                "best epoch": best_epoch,
            })

    if config.dataset == "Synapse":
        assert modified_dice is not None
        assert modified_hd is not None
        assert previous_dice is not None
        assert previous_hd is not None
        
        wandb.log({ "last modified test mean dice": modified_dice,
                    "last modified test mean hd95": modified_hd,
                    "last previous test mean dice": previous_dice,
                    "last previous test mean hd95": previous_hd,})

    torch.save(num_logger, os.path.join(config.results_dir, 'num_logger.pt'))  # type: ignore
    wandb.finish()
    return "Training Finished!"

# train for one epoch
def train_epoch(model, data_loader, optimizer, epoch, config, num_logger, iter_num=0):
    model.train()
    
    total_loss, total_loss_ce, total_loss_dice, total_reg_dac, total_num, train_bar = 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    description = ''
    for batch, (smp1, smp2) in enumerate(train_bar):
        x1, y1, idx1 = smp1['image'].cuda(), smp1['label'].cuda(), smp1['idx'].cuda()
        if config.data_choice == 'single':
            x2 = y2 = idx2 = None 
        else:
            x2, y2, idx2 = smp2['image'].cuda(), smp2['label'].cuda(), smp2['idx'].cuda()

        loss, loss_ce, loss_dice, reg_dac, weights, logits1, batch_num = model(x1, y1, idx1, x2, y2, idx2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = adjust_learning_rate(optimizer, epoch-1, config, iter_num, batch)
        iter_num += 1

        total_num += batch_num
        total_loss += loss.item() * batch_num
        total_loss_ce += torch.sum(loss_ce).item() 
        total_loss_dice += torch.sum(loss_dice).item() 
        total_reg_dac += torch.sum(reg_dac).item() 
        description = 'Epoch:[{}/{}], lr:{:.4f}, Loss:{:.4f}, CE:{:.4f}, DSC:{:.4f}, Reg:{:.4f}'.format(
                        epoch, config.epochs, 
                        optimizer.param_groups[0]['lr'], 
                        total_loss/total_num, 
                        total_loss_ce/total_num, 
                        total_loss_dice/total_num, 
                        total_reg_dac/total_num,
                        )
        train_bar.set_description(description)
        
        num_logger['ce'][idx1,epoch-1] = loss_ce
        num_logger['dice'][idx1,epoch-1] = loss_dice 
        num_logger['dac'][idx1,epoch-1] = reg_dac
        num_logger['w_dac'][idx1,epoch-1] = weights[:,1]

        wandb.log({ "total_loss": loss.item(),
                    "loss_ce": torch.mean(loss_ce).item(),
                    "loss_dice": torch.mean(loss_dice).item(),
                    "reg_dac": torch.mean(reg_dac).item(),
                    "lr": lr_,
                    "batch_num": batch_num
                })
    
    logging.info(description)
    return iter_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, config, iter_num, batch):
    """Decay the learning rate based on schedule"""
    lr = config.lr
    if config.decay_lr == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config.epochs))
    elif config.decay_lr == 'poly':  
        lr = config.lr * (1.0 - iter_num / config.max_iterations) ** 0.9
    elif config.decay_lr == 'magic':
        lr =  0.2 * config.batch_size / 256
        lr *= (1.0 - (epoch * config.batch_size + batch) / config.max_iterations) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# validation
def valid_epoch(model, config, valloader, db_val, best_performance, epoch, best_epoch):
    model.eval()
    metric_list = 0.0
    for _, (sampled_batch, _) in enumerate(valloader):
        image, label = sampled_batch["image"], sampled_batch["label"]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=config.num_classes,
            patch_size=[config.img_size, config.img_size],
        )
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_val)
    for class_i in range(config.num_classes - 1):
        wandb.log(
            {
                f"val class {class_i + 1} mean dice": metric_list[class_i, 0], # type: ignore
                f"val class {class_i + 1} mean hd95": metric_list[class_i, 1], # type: ignore
            }
        )

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    wandb.log(
        {
            "val_mean_dice": performance,
            "val_mean_hd95": mean_hd95,
        }
    )

    if performance > best_performance:
        best_epoch, best_performance, best_hd95 = (
            epoch,
            performance,
            mean_hd95,
        )
        save_best = os.path.join(config.results_dir, "best_model.pth")
        torch.save(model.state_dict(), save_best)
        logging.info(
            "Best model | iteration %d : mean_dice : %f mean_hd95 : %f"
            % (epoch, performance, mean_hd95)
        )

    logging.info(
        "iteration %d : mean_dice : %f mean_hd95 : %f"
        % (epoch, performance, mean_hd95)
    )
    model.train()
    return best_performance, best_epoch


# test
@torch.no_grad()
def inference(model, config, metric_choice="modified"):
    # dataloader
    testloader, db_test = get_test_loader(config)

    # logging
    logging.info("\n{} test iterations per epoch".format(len(testloader)))
    if config.test_save_path != '' and not os.path.exists(config.test_save_path):
        os.makedirs(config.test_save_path)

    # testing
    model.eval()
    metric_list = 0.0
    for i_batch, (sampled_batch, _) in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume( image,
                                        label,
                                        model,
                                        classes=config.num_classes,
                                        patch_size=[config.img_size, config.img_size],
                                        test_save_path=config.test_save_path,
                                        case=case_name,
                                        z_spacing=config.z_spacing,
                                        metric_choice=metric_choice,
        )
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, config.num_classes):
        logging.info('Mean CE %d mean_dice %f mean_hd95 %f' %
                     (i, metric_list[i-1][0], metric_list[i-1][1])) # type: ignore
        wandb.log({ f"{metric_choice} test class {i} mean dice": metric_list[i-1][0], # type: ignore
                    f"{metric_choice} test class {i} mean hd95": metric_list[i-1][1], # type: ignore
                })
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model with %s: mean_dice : %f mean_hd95 : %f \n' % (metric_choice, performance, mean_hd95))
    wandb.log({ f"{metric_choice} test mean dice": performance,
                f"{metric_choice} test mean hd95": mean_hd95,
            })
    return performance, mean_hd95


if __name__=='__main__':
    train()
    