#!/bin/bash
# bigcity_render.sh - Rendering script for MatrixCity BigCity trained model

# Check arguments


model_path="/home/hexu/clm_release/Grendel-XS/output/bigcity/20251112_223952_bigcity_102m_clm_offload_nosparseadam"
load_pt_path="/home/hexu/clm_release/Grendel-XS/output/bigcity/20251112_223952_bigcity_102m_clm_offload_nosparseadam/saved_tensors/iteration_499969"

# Validate that model path exists
if [ ! -d "$model_path" ]; then
    echo "Error: Model path does not exist: $model_path"
    exit 1
fi


echo "Model path: $model_path"
echo "Offload strategy: clm_offload (matching training)"

# Get the dataset path from the training config if available
if [ -f "$model_path/cfg_args" ]; then
    echo ""
    echo "Training configuration found in $model_path/cfg_args"
fi

# Offload configuration - MUST match training settings
# Pre-allocate for 102M+ Gaussians (same as training)
OFFLOAD_OPTS="--clm_offload \
--adam_type cpu_adam \
--prealloc_capacity 102_231_360 \
--sparse_adam \
--grid_size_D 128 \
--fused_adam torch_fused"

# Monitoring settings
MONITOR_OPTS="--enable_timer \
--check_gpu_memory \
--check_cpu_memory"

# Configure CUDA caching allocator (same as training)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "========================================"
echo "Rendering MatrixCity BigCity (Extreme Scale)"
echo "========================================"
echo "Model: $model_path"
echo "Strategy: clm_offload"
echo "Pre-allocated capacity: 102,231,360 Gaussians"
echo "========================================"
echo ""

# Run rendering
# Note: The model_path argument for render.py should point to the parent folder
# and iteration is specified separately
python render.py \
    -m ${model_path} \
    ${OFFLOAD_OPTS} \
    ${MONITOR_OPTS} \
    --load_pt_path ${load_pt_path} \
    --generate_num 30000 \
    --max_frames_per_video 3000 \
    --render_video \
    --video_only

# Check if rendering succeeded
if [ $? -ne 0 ]; then
    echo "Error: Rendering failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Rendering completed successfully!"
echo "========================================"
echo "Train renders: ${model_path}/train/ours_${iteration}/renders"
echo "Train GT:      ${model_path}/train/ours_${iteration}/gt"
echo "Test renders:  ${model_path}/test/ours_${iteration}/renders"
echo "Test GT:       ${model_path}/test/ours_${iteration}/gt"
echo "========================================"
echo ""
echo "To render only test set (faster):"
echo "  bash bigcity_render.sh $model_path $iteration --skip_train"
echo ""
echo "To render only train set:"
echo "  bash bigcity_render.sh $model_path $iteration --skip_test"