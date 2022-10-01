# AdaWAC: Adaptively Weighted Augmentation Consistency Regularization for Volumetric Medical Image Segmentation
This repository is the official implementation of _AdaWAC: Adaptively Weighted Augmentation Consistency Regularization for Volumetric Medical Image Segmentation_


## File Organization
To run AdaWAC, the relative paths for the datasets, the pre-trained models, and the pre-allocated directory for results are configured as follows.

```python
.
|-- adawac # main implementation 
|-- data # Synapse and ACDC datasets
|   |-- ACDC
|   |   |-- ACDC_training_slices
|           |-- patient*_frame*_slice_*.h5
|   |   |-- ACDC_training_volumes
|           |-- patient*_frame*.h5
|   |-- Synapse
|       |-- test_vol_h5
|           |-- case*.npy.h5
|       |-- train_npz
|           |-- case*_slice*.npz
|-- model # pretrained models
|   |-- vit_checkpoint
|       |-- imagenet21k
|           |-- R50+ViT-B_16.npz
|           |-- ViT-B_16.npz
|-results # experiment results
```

We followed the [official implementation](https://github.com/Beckschen/TransUNet) of [TransUNet](https://arxiv.org/abs/2102.04306) for data and pre-trained model preparation.


## Requirements
```
pip install -r requirements.txt
```


## Reproducing Main Experiments:
```python
bash reproduce.sh
```


## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)