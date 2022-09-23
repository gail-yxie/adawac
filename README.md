# Adaptively Weighted Augmentation Consistency (AdaWAC)

## File Organization
The main implementation in `adawac` is attached in the supplementary materials and will be published on GitHub.

To run AdaWAC, the relative paths for the datasets. pretrained models, and results are configured as follows.

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

## Environment Setup
```
pip install -r requirements.txt
```

## Reproducing Experiments:
```python
bash runs/train_subset.sh # sample efficiency and robustness of AdaWAC
bash runs/trim_loss.sh # comparisons with hard thresholding algorithms 
bash runs/ablation.sh # ablation study
```