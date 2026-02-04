CUDA_VISIBLE_DEVICES=0 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/100pct \
     --clm_offload \
     --sparse_adam \
     --bsz 4 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 5000 10000 15000 20000 25000 30000


CUDA_VISIBLE_DEVICES=1 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/100pct16 \
     --clm_offload \
     --sparse_adam \
     --bsz 16 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 5000 10000 15000 20000 25000 30000


CUDA_VISIBLE_DEVICES=3 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/100pct16 \
     --clm_offload \
     --sparse_adam \
     --bsz 16 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 2000 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000 25000 30000 \
     --save_iterations 29000 \
     --auto_start_checkpoint


CUDA_VISIBLE_DEVICES=3 python train.py \
     --source_path /data/jiajun/grid/Images/images_100pct/undistord \
     --model_path /data/jiajun/grid/output/100pct16 \
     --clm_offload \
     --sparse_adam \
     --bsz 16 \
     --iterations 30000 \
     --prealloc_capacity 1000000 \
     --checkpoint_iterations 2000 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000 25000 30000 \
     --save_iterations 29000 \
     --start_checkpoint /data/jiajun/grid/output/100pct16/checkpoints/24993/