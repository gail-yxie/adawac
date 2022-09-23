#!/bin/bash

# DATASET=Synapse
# EPOCHS=150
MODEL=unet
DICE_RATIO=0.5
# LOSS=adawac
# DECAY=no_decay
DAC_EN=1.0
LR_W=1.0

DECAY=poly
LOSS=base

for LR in 0.05 0.01;
do  
    DATASET=Synapse
    EPOCHS=150
    # CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --decay-lr $DECAY --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark unet-adawac_$LR_W

    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --decay-lr $DECAY --model $MODEL --dice-ratio $DICE_RATIO --loss $LOSS --exp-mark unet-base_$LR --data-choice single

    DATASET=ACDC
    EPOCHS=360

     CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --decay-lr $DECAY --model $MODEL --dice-ratio $DICE_RATIO --loss $LOSS --exp-mark unet-base_$LR --data-choice single

    # CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --decay-lr $DECAY --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark unet-adawac_$LR_W
done