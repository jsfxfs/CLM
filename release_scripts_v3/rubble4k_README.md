# CLM-GS Example Scripts: Rubble 4K

This script demonstrates large-scale real-world scene reconstruction using the Rubble dataset.

**Note:** Launch all scripts from the base folder of this repository (where `train.py` is located).

## Download the Rubble Dataset

You can download the Rubble dataset from the Mega-NeRF project: https://github.com/cmusatyalab/mega-nerf

### Dataset Preparation

We provide a preprocessed version of the Rubble dataset in COLMAP format. Download it from: [TODO: Add download link]

The expected dataset structure should contain standard COLMAP outputs (cameras, images, points3D) with images in the `images` folder.

## Running Experiments

The script accepts three parameters:
1. **dataset_folder**: Path to the Rubble dataset
2. **offload_strategy**: One of `no_offload`, `clm_offload`, or `naive_offload`
3. **scale**: Either `10m` (10 million Gaussians) or `28m` (28 million Gaussians)

The scale parameter controls the aggressiveness of the densification process, resulting in different numbers of Gaussians.

### Command Syntax

```bash
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh <dataset_folder> <offload_strategy> <scale>
```

### Example Commands

```bash
# 10M Gaussians experiments (less aggressive densification)
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm no_offload 10m
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm clm_offload 10m      # Recommended
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm naive_offload 10m

# 28M Gaussians experiments (more aggressive densification)
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm no_offload 28m
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm clm_offload 28m      # Recommended
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/rubble4k.sh /path/to/rubble-pixsfm naive_offload 28m
```

**Note:** The `clm_offload` strategy is recommended for memory efficiency, especially for the 28M scale.

### Output

Training results will be saved to `output/rubble/<timestamp>_rubble4k_<scale>_<strategy>/`.

### Analyzing Results

After all experiments finish, you can aggregate results using:

```bash
python release_scripts_v3/log2csv.py output/rubble
```

This will generate a CSV summary of PSNR, training time, and memory usage across all experiments.

## Expected Results

The experiments report the following metrics for each configuration:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Training time**
- **Maximum GPU memory usage**
- **Maximum CPU memory usage**

The `clm_offload` strategy demonstrates significant memory efficiency while maintaining rendering quality.

## Notes on Hyperparameters

### Densification Settings

- **10M scale**: Uses more conservative densification parameters
  - Gradient threshold: 0.00015
  - Percent dense: 0.005
  - Opacity reset interval: 3000
  - Densify from iteration 4000 to 21000

- **28M scale**: Uses more aggressive densification parameters
  - Gradient threshold: 0.0001
  - Percent dense: 0.002
  - Opacity reset interval: 9000
  - Densify from iteration 5000 to 21000

### Pre-allocated Capacity

For `clm_offload`, the script pre-allocates 30M capacity in CPU pinned memory. If your system has limited CPU memory (< 16GB), you may need to:
1. Use the 10M scale with less aggressive densification
2. Reduce the `--prealloc_capacity` parameter in the script

### Memory Requirements

- **No offload**: Requires sufficient GPU memory to hold all Gaussians (may OOM for 28M)
- **Naive offload**: Offloads to CPU but with higher overhead
- **CLM offload**: Most memory-efficient, recommended for large scales
