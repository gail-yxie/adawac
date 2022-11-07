#!/bin/bash

MODEL=unet
DATASET=Synapse
EPOCHS=150
DICE_RATIO=0.5
DAC_EN=2.0
LR_W=2.0
LR=0.01

# baseline
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO 
#--decay-lr poly

# AdaWAC
for DAC_EN in 1.0 2.0 4.0;
do
    for LR_W in 2.0 4.0;
    do
        CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W #--decay-lr poly
    done
done

# # Pseudo-AdaWAC
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss pseudo --dac-encoder $DAC_EN --exp-mark pseudo

# # DATASET=ACDC
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs 360 --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W

