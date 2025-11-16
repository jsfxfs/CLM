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

<!-- CUDA_VISIBLE_DEVICES=1 bash release_scripts_v3/bigcity.sh /mnt/nvme0/dataset/matrixcity/big_city/aerial/pose/all_blocks 25m -->

**Note:** `no_offload` and `naive_offload` strategies typically result in out-of-memory (OOM) errors for 100M scale. Only `clm_offload` can successfully handle extreme scales.

### Output

Training results will be saved to `output/bigcity/<timestamp>_bigcity_<scale>_<strategy>/`.

### Analyzing Results

After all experiments finish, you can aggregate results using:

```bash
python release_scripts_v3/log2csv.py output/bigcity
```

This will generate a CSV summary of PSNR, training time, and memory usage across all experiments.

## Experiments Results on our testbed

The experiments report the following metrics for each configuration:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Training time**
- **Maximum GPU memory usage**
- **Pinned CPU Memory (GB)**

The `clm_offload` strategy demonstrates significant memory efficiency while maintaining rendering quality, enabling training at scales that would otherwise be impossible on standard hardware. 

<!-- python bigcitypostprocess.py /home/hexu/clm_release/Grendel-XS/output/bigcity/experiment_results.csv -->

| Experiment                    | Test PSNR   | Train PSNR   | Num 3DGS   | Max GPU Memory (GB)   | Pinned CPU Memory (GB)   | Training Time (s)   |
|:------------------------------|:------------|:-------------|:-----------|:----------------------|:-------------------------|:--------------------|
| bigcity_100m_clm_offload_102M | 25.5        | 26.84        | 102231360  | 20.79                 | 37.41                    | 11783.36            |
| bigcity_102M_naive_offload    | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
| bigcity_102M_no_offload       | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
| bigcity_25m_clm_offload_25M   | 24.63       | 25.9         | 25557840   | 5.64                  | 10.04                    | 6029.05             |
| bigcity_25m_naive_offload_25M | 24.29       | 25.12        | 25557840   | 12.13                 | 19.87                    | 10187.07            |
| bigcity_25M_no_offload        | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
