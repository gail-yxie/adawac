#!/bin/bash

DATASET=Synapse
MODEL=transunet
EPOCHS=150
DICE_RATIO=0.5
LR=0.05
DECAY=no_decay
# LR_W=1.0
LOSS=adawac

# for DAC_EN in 0.5 2.0 0.2 5.0;
# do
#     CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss $LOSS --dac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$DAC_EN
# done

DAC_EN=0.5

for LR_W in 0.01 0.02 0.05 0.1 0.2 0.5 2.0;
do
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss $LOSS --dac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_LRW-$LR_W-DAC-$DAC_EN
done
