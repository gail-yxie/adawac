# AdaWAC: Adaptively Weighted Augmentation Consistency Regularization for Volumetric Medical Image Segmentation

This repository is the official implementation of _AdaWAC: Adaptively Weighted Augmentation Consistency Regularization for Volumetric Medical Image Segmentation_

## File Organization
The main implementation in `adawac` is attached in the supplementary materials.

To run AdaWAC, the relative paths for the datasets. pretrained models and results are configured as follows.

```python
..
|-adawac # main implementation 
|-data # Synapse and ACDC datasets
| |-ACDC
| | |-ACDC_training_slices
| | |-ACDC_training_volumes
| |-Synapse
| | |-test_vol_h5
| | |-train_npz
|-model # pretrained models
| |-vit_checkpoint
| | |-imagenet21k
| | | |-R50+ViT-B_16.npz
| | | |-ViT-B_16.npz
|-results # experiment results
```

## Requirements
```
pip install -r requirements.txt
```

## Reproducing Main Experiments:
```python
bash reproduce.sh
```