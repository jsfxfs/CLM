# CLM-GS Example Scripts: MatrixCity BigCity Aerial View

This script demonstrates the extreme-scale capability of the Grendel-XS system using the MatrixCity BigCity dataset.

**Note:** Launch all scripts from the base folder of this repository (where `train.py` is located).
**Note:** We start from very large dataset and disable the densification in this bigcity experiment. 


## Download the MatrixCity Dataset

(FIXME) We provide instruction of how to prepare the big city dataset. You need to ensure the folder path contains the key word `matrixcity` to be detected as matrixcity dataset. We will download our preprocessed point cloud here: xxx

The dataset includes both a small city and a big city. **This script uses the big city aerial view.**

### Dataset Preparation

Ensure your dataset folder path contains the keyword `matrixcity` to be correctly detected. The expected structure is:

```
/path/to/matrixcity/big_city/aerial/pose/all_blocks/
```

## Running Experiments

The script accepts three parameters:
1. **dataset_folder**: Path to the MatrixCity BigCity dataset
2. **offload_strategy**: One of `no_offload`, `clm_offload`, or `naive_offload`
3. **scale**: Either `100m` (100 million Gaussians) or `25m` (25 million Gaussians)

### Command Syntax

```bash
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh <dataset_folder> <offload_strategy> <scale>
```

### Example Commands

```bash
# 100M Gaussians experiments
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks no_offload 100m      # Expected: OOM
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 100m     # Recommended
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks naive_offload 100m   # Expected: OOM

# 25M Gaussians experiments
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks no_offload 25m       # Expected: OOM
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 25m      # Recommended
CUDA_VISIBLE_DEVICES=0 bash release_scripts_v3/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks naive_offload 25m
```

**Note:** `no_offload` and `naive_offload` strategies typically result in out-of-memory (OOM) errors for 100M scale. Only `clm_offload` can successfully handle extreme scales.

### Output

Training results will be saved to `output/bigcity/<timestamp>_bigcity_<scale>_<strategy>/`.

### Analyzing Results

After all experiments finish, you can aggregate results using:

```bash
python release_scripts_v3/log2csv.py output/bigcity
```

This will generate a CSV summary of PSNR, training time, and memory usage across all experiments.

## Expected Results

The experiments report the following metrics for each configuration:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Training time**
- **Maximum GPU memory usage**
- **Maximum CPU memory usage**

The `clm_offload` strategy demonstrates significant memory efficiency while maintaining rendering quality, enabling training at scales that would otherwise be impossible on standard hardware. 
