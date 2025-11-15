# CLM-GS Example Scripts: mip360

NOTE: you should launch these scripts in the base folder of this repository (where train.py is located). 
We call it `path/to/clm-gs`. 

NOTE: mip360 is a very small scene, and it does not require cpu offloading at all, GPU memory should be enough. So this set of experiment is just for sanity check of PSNR is the same. 

## Download the mip360 dataset

Download the dataset from https://jonbarron.info/mipnerf360/ . 
You can simply run: wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

You should unzip the dataset into `/path/to/360_v2`. 

## Run the experiments

You need to specify the folder storing the mip360 dataset; and specify the mode for the training. 

This will run the scenes of counter bicycle stump garden room bonsai kitchen all together. 

```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 naive_offload
```

The training results will be saved into `path/to/clm-gs/output/mip360`. 

After all experiment finishes, you can also run `python log2csv.py path/to/clm-gs/output/mip360` 


<!-- ```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 naive_offload
``` -->

## Notes on Hyperparameters

1. **Pre-allocation Capacity**: `--prealloc_capacity 7_000_000` specifies the number of Gaussians to pre-allocate in CPU pinned memory. If you have limited CPU memory (< 8GB), you may encounter out-of-memory errors. In this case, reduce densification aggressiveness and lower the `--prealloc_capacity` value accordingly.

2. **Batch Size**: We use `--bsz 4` for all scenes in these experiments. 

## PSNR Comparison by Scene and Offload Type

We compare the psnr results for all scenes between three modes. 

## Test PSNR

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |         25.24 |           25.24 |        25.22 |
| bonsai  |         32.11 |           31.81 |        32.1  |
| counter |         29.07 |           29.08 |        28.97 |
| garden  |         27.36 |           27.35 |        27.31 |
| kitchen |         31.53 |           31.48 |        31.4  |
| room    |         31.39 |           31.45 |        31.46 |
| stump   |         26.7  |           26.68 |        26.62 |

## Train PSNR

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |         26.38 |           26.37 |        26.23 |
| bonsai  |         33.26 |           33.26 |        33.43 |
| counter |         30.5  |           30.65 |        30.47 |
| garden  |         29.8  |           29.85 |        29.72 |
| kitchen |         33.15 |           32.93 |        32.77 |
| room    |         34.17 |           34.21 |        34.18 |
| stump   |         30.69 |           31.09 |        30.63 |
