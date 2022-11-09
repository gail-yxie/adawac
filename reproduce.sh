#!/bin/bash

# DATASET=Synapse
# EPOCHS=150
# LR=0.05
# MODEL=transunet
# DICE_RATIO=0.5
# LOSS=adawac
# DAC_EN=2.0
# LR_W=2.0
# LR=0.01

# # ERM
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL
# # Pseudo-AdaWAC
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss pseudo --dac-encoder $DAC_EN --exp-mark pseudo
# # AdaWAC
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W

# # DATASET=ACDC
# # EPOCHS=360
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs $EPOCHS --lr $LR --model $MODEL --dice-ratio $DICE_RATIO --dac --loss $LOSS --dac-encoder $DAC_EN --lr-w $LR_W --exp-mark adawac_$LR_W


MODEL=transunet
SEED=1234

DATASET=Synapse
EPOCHS=150
LR=0.01
DAC_EN=2.0
LR_W=2.0

# partial dataset
for PARTIAL in half_slice_sparse;
do
    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs 300 --model $MODEL --lr $LR --exp-mark base-$PARTIAL-LRW$LR_W --partial $PARTIAL --test_save_path ../test/ --seed $SEED

    CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --epochs 300 --model $MODEL --lr $LR --loss adawac --exp-mark adawac-$PARTIAL-LRW$LR_W --partial $PARTIAL --dac --dac-encoder $DAC_EN --lr-w $LR_W --test_save_path ../test/ --seed $SEED
done