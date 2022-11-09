#!/bin/bash

MODEL=unet
SEED=1234

DATASET=Synapse
EPOCHS=150
DAC_EN=2.0
LR_W=2.0
LR=0.01

# Baseline
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --seed $SEED

for DAC_EN in 2.0 4.0;
do 
    for LR_W in 4.0 8.0;
    do
        # AdaWAC
        CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss adawac --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W --seed $SEED
    done

    # Pseudo-AdaWAC
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dac --loss pseudo --dac-encoder $DAC_EN --exp-mark pseudo --seed $SEED
done

