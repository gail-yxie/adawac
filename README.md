# AdaWAC: Adaptively Weighted Data Augmentation Consistency Regularization for Robust Optimization under Concept Shift
This repository presents code for our [paper](https://arxiv.org/abs/2210.01891):
```
Adaptively Weighted Data Augmentation Consistency Regularization for Robust Optimization under Concept Shift.
Yijun Dong*, Yuege Xie*, Rachel Ward. ICML 2023.
```


## File Organization
To run AdaWAC, the relative paths for the datasets, the pre-trained models, and the pre-allocated directory for results are configured as follows.

```python
.
|-- adawac # main implementation 
|-- data # Synapse and ACDC datasets
|   |-- ACDC
|   |   |-- ACDC_training_slices
|   |   |   |-- patient*_frame*_slice_*.h5
|   |   |-- ACDC_training_volumes
|   |       |-- patient*_frame*.h5
|   |-- Synapse
|       |-- test_vol_h5
|       |   |-- case*.npy.h5
|       |-- train_npz
|           |-- case*_slice*.npz
|-- model # pretrained models
|   |-- vit_checkpoint
|       |-- imagenet21k
|           |-- R50+ViT-B_16.npz
|           |-- ViT-B_16.npz
|-- results # experiment results
```

We followed the [official implementation](https://github.com/Beckschen/TransUNet) of [TransUNet](https://arxiv.org/abs/2102.04306) for data and pre-trained model preparation.


## Requirements
```
pip install -r requirements.txt
```


## Reproducing Main Experiments:
```python
bash run_transunet.sh
bash run_unet.sh
```


## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)
