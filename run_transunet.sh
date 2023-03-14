#!/bin/bash

MODEL=transunet
SEED=1234

DATASET=Synapse
EPOCHS=150
LR=0.01
DAC_EN=2.0
LR_W=2.0

# main: ERM v.s. AdaWAC
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
# ERM
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED
# Pseudo-AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss pseudo --dac-encoder $DAC_EN --seed $SEED

# # partial dataset
# for PARTIAL in half_vol half_slice;
# do
#     CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --model $MODEL --lr $LR --partial $PARTIAL --seed $SEED

#     CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --model $MODEL --lr $LR --loss adawac --partial $PARTIAL --dac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
# done

# hard thresholding 
# trim-ratio
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-ratio --seed $SEED
# trim-ratio + ACR
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-ratio --dac --dac-encoder $DAC_EN --seed $SEED
# trim-train
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-train --seed $SEED
# trim-train + ACR
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-train --dac --dac-encoder $DAC_EN --seed $SEED

# Ablation
# reweight-only
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss reweight-only --seed $SEED
# ARC-only
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss dac-only --dac --dac-encoder $DAC_EN --seed $SEED
# adawac
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss adawac --dac --dac-encoder $DAC_EN --lr-w 0.02 --seed $SEED

# ACDC
DATASET=ACDC
EPOCHS=360
# ERM
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
