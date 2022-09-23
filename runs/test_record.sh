#!/bin/bash

# DATASET=ACDC
MODEL=transunet
# EPOCHS=360
DICE_RATIO=0.5
LR=0.05
DECAY=no_decay
DAC_EN=1.0
LOSS=adawac
LR_W=1.0

# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --dac --dac-encoder $DAC_EN --loss $LOSS --lr-w $LR_W --exp-mark adawac-dac$DAC_EN-lrw$LR_W --test_save_path ../test_new/$DATASET/adawac-dac$DAC_EN-lrw$LR_W

DATASET=Synapse
EPOCHS=150

CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --dac --dac-encoder $DAC_EN --loss $LOSS --lr-w $LR_W --exp-mark adawac-dac$DAC_EN-lrw$LR_W-back --test_save_path ../test_new/$DATASET/adawac-new-dac$DAC_EN-lrw$LR_W-back

# for base Synapse
DECAY=poly
LOSS=base
CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss $LOSS --exp-mark base_$LR-back --data-choice single --test_save_path ../test_new/$DATASET/base_$LR-back

# for base ACDC
# DATASET=ACDC
# EPOCHS=360
# CUDA_VISIBLE_DEVICES=0 python train_dac.py --dataset $DATASET --model $MODEL --epochs $EPOCHS --dice-ratio $DICE_RATIO --lr $LR --decay-lr $DECAY --loss $LOSS --exp-mark base_$LR --data-choice single --test_save_path ../test_new/$DATASET/base_$LR
