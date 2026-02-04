CUDA_VISIBLE_DEVICES=1 python render_bigcity_images.py \
    --source_path /data/jiajun/grid/Images/images_100pct/undistord \
    --model_path /data/jiajun/grid/output/depth1 \
    --iteration 25985 \
    --traj_path ellipse \
    --n_frames 240 \
    --clm_offload \
    --save_video

CUDA_VISIBLE_DEVICES=1 python render_bigcity_images.py \
    --source_path /data/jiajun/grid/Images/images_100pct/undistord \
    --model_path /data/jiajun/grid/output/depth1 \
    --iteration 25985 \
    --traj_path ellipse \
    --n_frames 10 \
    --traj_path original  \
    --clm_offload
