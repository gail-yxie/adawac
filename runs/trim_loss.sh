#!/bin/bash

DATASET=Synapse
MODEL=transunet
EPOCHS=150
DICE_RATIO=0.5
LR=0.05
DECAY=no_decay
DAC_EN=0.5

# # trim-ratio
# for TRIM_RATIO in 0.0 0.1 0.3 0.5 0.9;
# do
#     CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss trim-ratio --trim-ratio $TRIM_RATIO --dac --dac-encoder $DAC_EN --exp-mark trim-ratio-$TRIM_RATIO-CR-$DAC_EN
# done

# trim-train
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss trim-train --dac --dac-encoder $DAC_EN --exp-mark trim-train-CR-$DAC_EN

# # trim-data
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss ce --partial trim --dac --dac-encoder $DAC_EN --exp-mark trim-data-CR-$DAC_EN

