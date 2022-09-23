#!/bin/bash

DATASET=Synapse
EPOCHS=300
MODEL=transunet
DICE_RATIO=0.5
LR=0.05
DECAY=no_decay
LOSS=adawac
DAC_EN=0.5
LR_W=1.0

for PARTIAL in half_slice half_vol half_slice_sparse half_vol_sparse;
do
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --model $MODEL --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY  --loss $LOSS --exp-mark adawac_$PARTIAL-$LR-LRW-$LR_W --partial $PARTIAL --dac --dac-encoder $DAC_EN --lr-w $LR_W
done
