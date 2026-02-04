CUDA_VISIBLE_DEVICES=1 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/depth2 \
     --clm_offload \
     --sparse_adam \
     --bsz 16 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 2000 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000 25000 30000 \
     --save_iterations 26000  \
     --start_checkpoint /data/jiajun/grid/output/depth2/checkpoints/24993/
     --llffhold 10 \
     --dense_ply_file /data1/home/jiajun/TEST/CLM-GS/FilteredPointcloud/lane2_density_filtered.ply

--auto_start_checkpoint \

CUDA_VISIBLE_DEVICES=0 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/depth1 \
     --clm_offload \
     --sparse_adam \
     --bsz 16 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 2000 4000 6000 8000 10000 15000 20000 25000 30000 \
     --start_checkpoint /data/jiajun/grid/output/depth1/checkpoints/24993/    \
     --save_iterations 26000  \
     --llffhold 10 \
     --dense_ply_file /data1/home/jiajun/TEST/CLM-GS/FilteredPointcloud/lane1_density_filtered.ply