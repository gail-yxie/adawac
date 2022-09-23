#!/bin/bash

DATASET=Synapse
MODEL=transunet
EPOCHS=150
DICE_RATIO=0.5
LR=0.05
DECAY=no_decay
# DAC_EN=1.0

# DAC only
DAC_EN=0.5
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --dac --dac-encoder $DAC_EN --exp-mark dac-only

# # reweight only
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss reweight-only --exp-mark reweight-only

# # reweight with other lr_w
# LR_W=0.01
# LOSS=adawac
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --dac --dac-encoder $DAC_EN --loss $LOSS --lr-w $LR_W --exp-mark adawac_$LR_W
