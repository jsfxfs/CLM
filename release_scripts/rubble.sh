#!/bin/bash

# rubble.sh - Training script for Rubble dataset
# This script demonstrates large-scale real-world scene reconstruction

# Check arguments
if [ $# -ne 2 ]; then
    echo "Error: Please specify exactly two arguments."
    echo "Usage: bash rubble.sh <dataset_folder> <offload_strategy>"
    echo ""
    echo "Arguments:"
    echo "  <dataset_folder>   : Path to the rubble dataset folder"
    echo "  <offload_strategy> : One of: no_offload, naive_offload, clm_offload"
    echo ""
    echo "Example:"
    echo "  bash rubble.sh /path/to/rubble-pixsfm clm_offload"
    exit 1
fi

dataset_folder=$1
offload_strategy=$2

echo "Dataset folder: $dataset_folder"
echo "Offload strategy: $offload_strategy"

# Validate offload strategy
if [ "$offload_strategy" != "no_offload" ] && [ "$offload_strategy" != "naive_offload" ] && [ "$offload_strategy" != "clm_offload" ]; then
    echo "Error: Invalid offload strategy '$offload_strategy'"
    echo "Must be one of: no_offload, naive_offload, clm_offload"
    exit 1
fi

# Generate timestamp for experiment naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Experiment name
expe_name="rubble_${offload_strategy}"

# Set output paths
log_folder="output/rubble/${TIMESTAMP}_${expe_name}"
model_path=${log_folder}

echo "Output folder: $log_folder"

# Downsample option (use images_4 for 4x downsampling)
DOWNSAMPLE_OPT="--images images_4"

# Training configurations
LLFFHOLD=83
BSZ=4
ITERATIONS=30000
LOG_INTERVAL=250

# Test and save iterations
TEST_ITERATIONS="7000 10000 15000 20000 25000 30000"
SAVE_ITERATIONS="7000 30000"

# Densification parameters
DENSIFY_OPTS="--densify_until_iter 15000 \
--densify_grad_threshold 0.0002 \
--percent_dense 0.01 \
--opacity_reset_interval 3000"

# Monitoring settings
MONITOR_OPTS="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Configure offload strategy
if [ "$offload_strategy" = "no_offload" ]; then
    offload_opts="--no_offload \
--fused_adam torch_fused"
elif [ "$offload_strategy" = "naive_offload" ]; then
    offload_opts="--naive_offload \
--adam_type cpu_adam \
--fused_adam torch_fused"
elif [ "$offload_strategy" = "clm_offload" ]; then
    offload_opts="--clm_offload \
--adam_type cpu_adam \
--prealloc_capacity 20_000_000 \
--grid_size_D 128 \
--fused_adam torch_fused"
fi

# Configure CUDA caching allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "========================================"
echo "Training Rubble Downsampled Scene"
echo "========================================"
echo "Strategy: $offload_strategy"
echo "Batch size: $BSZ"
echo "Iterations: $ITERATIONS"
echo "========================================"
echo ""

# Run training
python train.py \
    -s ${dataset_folder} \
    ${DOWNSAMPLE_OPT} \
    --llffhold ${LLFFHOLD} \
    --log_folder ${log_folder} \
    --model_path ${model_path} \
    --iterations ${ITERATIONS} \
    --log_interval ${LOG_INTERVAL} \
    --bsz ${BSZ} \
    --test_iterations ${TEST_ITERATIONS} \
    --save_iterations ${SAVE_ITERATIONS} \
    ${DENSIFY_OPTS} \
    ${offload_opts} \
    ${MONITOR_OPTS} \
    --eval

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "Results saved in: ${log_folder}"
echo "========================================"

