#!/bin/bash

MODEL=transunet
SEED=1234

DATASET=Synapse
EPOCHS=150
LR=0.01
DAC_EN=2.0
LR_W=2.0

# main: ERM v.s. AdaWAC
# ERM
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W --seed $SEED

# partial dataset
for PARTIAL in half_slice half_vol half_slice_sparse;
do
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs 300 --model $MODEL --lr $LR --exp-mark base-$PARTIAL-LRW$LR_W --partial $PARTIAL --seed $SEED

    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs 300 --model $MODEL --lr $LR --loss adawac --exp-mark adawac-$PARTIAL-LRW$LR_W --partial $PARTIAL --dac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED
done

# hard thresholding 
# Pseudo-AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss pseudo --dac-encoder $DAC_EN --exp-mark pseudo-adawac --seed $SEED
# trim-ratio
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-ratio --exp-mark trim-ratio-$TRIM_RATIO --seed $SEED
# trim-ratio + ACR
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-ratio --dac --dac-encoder $DAC_EN --exp-mark trim-ratio-$TRIM_RATIO-CR-$DAC_EN --seed $SEED
# trim-train
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-train --exp-mark trim-train --seed $SEED
# trim-train + ACR
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss trim-train --dac --dac-encoder $DAC_EN --exp-mark trim-train-CR-$DAC_EN --seed $SEED

# ACDC
DATASET=ACDC
EPOCHS=360
# ERM
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED
# AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --seed $SEED

# Ablation
# reweight-only
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss reweight-only --exp-mark reweight-only --seed $SEED
# ARC-only
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss dac-only --dac --dac-encoder $DAC_EN --exp-mark dac-only --seed $SEED
# adawac
LR_W = 0.02
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --lr $LR --loss adawac --dac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W --seed $SEED

