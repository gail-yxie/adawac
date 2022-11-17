#!/bin/bash

MODEL=unet
SEED=1234

DATASET=Synapse
EPOCHS=150
DAC_EN=2.0
LR_W=8.0
LR=0.01

# main: ERM v.s. AdaWAC
# ERM
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
# Pseudo-AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss pseudo --dac-encoder $DAC_EN --seed $SEED

# partial dataset
for PARTIAL in half_vol half_slice half_slice_sparse;
do
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --model $MODEL --lr $LR --partial $PARTIAL --seed $SEED

    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --model $MODEL --lr $LR --loss adawac --partial $PARTIAL --dac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
done