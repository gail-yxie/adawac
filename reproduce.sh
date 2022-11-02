#!/bin/bash

# DATASET=Synapse
# EPOCHS=150
# LR=0.05
#### already change it to separate way with masks
MODEL=transunet
DICE_RATIO=0.5
LOSS=adawac
DAC_EN=2.0
LR_W=2.0
LR=0.01

DATASET=Synapse
EPOCHS=150
# Pseudo-AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss pseudo --dac-encoder $DAC_EN --exp-mark pseudo
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W

# DATASET=ACDC
# EPOCHS=360
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W

