#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
CPU-Offloaded Image Rendering for 3D Gaussian Splatting

This script performs image rendering with CPU offloading using trained 3D Gaussian Splatting
models saved as PLY files. It enables rendering of large-scale scenes that may not fit entirely
in GPU memory by keeping Gaussian parameters in CPU RAM and loading them to GPU on-demand.

=== OVERVIEW ===
This script loads a trained 3D Gaussian Splatting model (saved as PLY point cloud files during
training) and renders individual images along camera trajectories. It's designed for
scenes with millions or billions of Gaussians where GPU memory is a constraint.

=== KEY FEATURES ===
- Single GPU rendering (no distributed training/inference setup required)
- Three offloading strategies:
  * clm_offload: Chunk-based loading with optimized memory management (recommended for large scenes)
  * naive_offload: Simple CPU↔GPU transfers (baseline offload strategy)
  * no_offload: Keep all data on GPU (fastest but highest memory usage)
- On-demand GPU loading: Transfers data to GPU only during active rendering
- Multiple trajectory types: interpolated, elliptical, spiral, or original training poses
- Individual image output: Saves each frame as a PNG image
- Optional video output: Can also generate MP4 videos at 30 FPS (use --save_video flag)

=== MODEL LOADING ===
IMPORTANT: This script loads models from PLY files (saved during training with scene.save()),
NOT from checkpoint files (.pth). Checkpoint files contain optimizer states and other training
data that are not needed for rendering.

Model files are expected at: {model_path}/point_cloud/iteration_{N}/point_cloud.ply

During training, models are saved using:
- scene.save(iteration) → saves as PLY files (standard format, used by this script)
- scene.save_tensors(iteration) → saves as .pt files (optional fast loading format)

=== USAGE EXAMPLE ===
# Render individual images with CLM offloading (recommended for large scenes):
python render_images.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration 30000 \\
    --traj_path ellipse \\
    --n_frames 240 \\
    --clm_offload

# Render images AND create a video:
python render_images.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration -1 \\
    --traj_path spiral \\
    --n_frames 120 \\
    --no_offload \\
    --save_video

# Use the render_single_image function in your own code:
from render_images import render_single_image
rendered_img = render_single_image(
    camera=my_camera,
    gaussians=my_gaussians,
    background=torch.tensor([1.0, 1.0, 1.0], device="cuda"),
    save_path="output/my_image.png",
    args=args,
    scene=None
)

=== TRAJECTORY TYPES ===
- "interp": Smooth interpolation between training camera poses (good for general use)
- "ellipse": Elliptical orbit around scene center at constant height (good for object-centric scenes)
- "spiral": 3D spiral path through scene bounds (good for scene exploration)
- "original": Use original training camera poses (good for validation/comparison)

=== TECHNICAL DETAILS ===
3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives. Each Gaussian has:
- Position: 3D coordinates (xyz) - 3 floats
- Rotation: Quaternion - 4 floats
- Scale: 3D anisotropic scaling - 3 floats
- Opacity: Transparency - 1 float
- Color: Spherical harmonics coefficients - (K coefficients × 3 RGB channels) floats

For large scenes (billions of Gaussians), storing all parameters on GPU exceeds available VRAM.
This script's offloading strategies enable rendering by:
1. Storing parameters in CPU RAM
2. Loading required data to GPU per frame (or per chunk)
3. Rendering the frame
4. Clearing GPU memory and moving to next frame

This allows rendering of arbitrarily large scenes on consumer GPUs.
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional, List
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    init_args,
)
from PIL import Image
from scene import Scene, load_scene_info_for_rendering
from scene.cameras import Camera
from strategies.naive_offload import GaussianModelNaiveOffload, naive_offload_eval_one_cam
from strategies.clm_offload import GaussianModelCLMOffload, clm_offload_eval_one_cam
from strategies.no_offload import GaussianModelNoOffload, baseline_accumGrads_micro_step

from traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)

from utils.general_utils import safe_state, PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from utils.graphics_utils import getWorld2View2
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import loadCam
import psutil
from scipy.spatial import ConvexHull

def create_camera_from_cam_info(
    cam_info,
    uid: int = 0,
):
    """
    Create a Camera object from CameraInfo with ground truth image loaded from disk.
    
    Args:
        cam_info: CameraInfo object containing camera parameters and image_path
        uid: Camera unique identifier
    
    Returns:
        Camera object with ground truth image loaded
    """
    # Get args and log file
    args = get_args()
    log_file = get_log_file()
    
    # Get resolution
    resolution = utils.get_img_width(), utils.get_img_height()
    
    # Load image from disk (following loadCam pattern)
    image = Image.open(cam_info.image_path)
    resized_image_rgb = PILtoTorch(
        image, resolution, args, log_file, decompressed_image=None
    )
    
    # Take first 3 channels and make contiguous
    gt_image = resized_image_rgb[:3, ...].contiguous()
    loaded_mask = None
    
    # Free the memory
    image.close()
    image = None
    
    camera = Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=uid,
        offload=True,  # Keep camera on CPU
    )

    # if the system cpu memory is low, exit
    if psutil.virtual_memory().percent > 90: # 90%
        print("System CPU memory is low, exiting...")
        exit()
    return camera

def create_camera_from_c2w(
    c2w: torch.Tensor,
    FoVx: float,
    FoVy: float,
    width: int,
    height: int,
    uid: int = 0,
    image_name: str = "trajectory_frame",
):
    """
    Create a Camera object from camera-to-world transform.
    
    Args:
        c2w: Camera-to-world transformation matrix [4, 4]
        FoVx: Horizontal field of view in radians
        FoVy: Vertical field of view in radians
        width: Image width
        height: Image height
        uid: Camera unique identifier
        image_name: Name for this camera/frame
    
    Returns:
        Camera object
    """
    # Convert c2w to R and T for Camera class
    c2w_np = c2w.cpu().numpy() if isinstance(c2w, torch.Tensor) else c2w
    
    # # Extract rotation matrix (first 3x3) and translation (last column)
    # R = c2w_np[:3, :3].T  # Camera uses world-to-camera rotation
    # T = -R @ c2w_np[:3, 3]  # Camera uses world-to-camera translation

    R = c2w_np[:3, :3]
    T = c2w_np[:3, 3]
    
    # Create dummy image (not needed for rendering)
    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
    
    camera = Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        offload=True,  # Keep camera on CPU
    )
    Camera.original_image_backup = None # delete the image

    # if the system cpu memory is low, delete the camera
    if psutil.virtual_memory().percent > 90: # 90%
        print("System CPU memory is low, exiting...")
        exit()
    return camera


def generate_trajectory_cameras(
    world_to_camera_all: list,
    camtoworlds_all: list,
    traj_path: str = "interp",
    n_frames: int = 120,
    FoVx: Optional[float] = None,
    FoVy: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """
    Generate camera trajectory for rendering.
    
    Args:
        world_to_camera_all: List of world-to-camera transformation matrices
        camtoworlds_all: List of camera-to-world transformation matrices
        traj_path: Type of trajectory ("interp", "ellipse", "spiral", "original")
        n_frames: Number of frames to generate (ignored for "original")
        FoVx, FoVy: Field of view (uses first training camera if None)
        width, height: Image size (uses first training camera if None)
    
    Returns:
        List of Camera objects for trajectory
    """
    # Check if we have training cameras
    if len(world_to_camera_all) == 0:
        raise ValueError("No training cameras found in scene")
    
    # stack camtoworlds_all to get train_c2w
    train_c2w = np.stack(camtoworlds_all, axis=0)
    # Generate trajectory based on type
    if traj_path == "original":
        print(f"Using original training poses: {len(train_c2w)} cameras")
        c2w_traj = train_c2w[:n_frames, :3, :] # only keep the first n_frames poses
    else:
        # Use subset of poses for trajectory generation (skip first/last few)
        train_c2w_subset = train_c2w[5:-5] if len(train_c2w) > 10 else train_c2w
        
        if traj_path == "interp":
            c2w_traj = generate_interpolated_path(
                train_c2w_subset[:, :3, :], n_interp=n_frames // len(train_c2w_subset)
            )
        elif traj_path == "ellipse":
            height_z = train_c2w_subset[:, 2, 3].mean()
            c2w_traj = generate_ellipse_path_z(
                train_c2w_subset[:, :3, :], n_frames=n_frames, height=height_z
            )
        elif traj_path == "spiral":
            # Compute bounds from training camera positions
            positions = train_c2w_subset[:, :3, 3]
            bounds = np.array([
                positions.min(axis=0) - 1.0,
                positions.max(axis=0) + 1.0
            ])
            c2w_traj = generate_spiral_path(
                train_c2w_subset[:, :3, :], bounds=bounds, n_frames=n_frames
            )
        else:
            raise ValueError(f"Unsupported trajectory type: {traj_path}")
    
    # Convert trajectory to homogeneous coordinates [N, 4, 4]
    c2w_traj_homo = np.concatenate([
        c2w_traj,
        np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(c2w_traj), axis=0),
    ], axis=1)
    
    # Create Camera objects for each pose
    # import pdb; pdb.set_trace()
    trajectory_cameras = []
    for idx, c2w in enumerate(c2w_traj_homo):
        camera = create_camera_from_c2w(
            c2w=torch.from_numpy(c2w).float(),
            FoVx=FoVx,
            FoVy=FoVy,
            width=width,
            height=height,
            uid=idx,
            image_name=f"traj_frame_{idx:05d}",
        )
        trajectory_cameras.append(camera)
    
    print(f"Generated trajectory with {len(trajectory_cameras)} cameras")
    return trajectory_cameras


def generate_convex_hull_trajectory(
    T_all: list,
    R_fixed: np.ndarray,
    height_z: float,
    n_frames: int,
    FoVx: float,
    FoVy: float,
    width: int,
    height: int,
):
    """
    Generate camera trajectory that follows the convex hull of camera positions.
    
    Args:
        T_all: List of all translation vectors from training cameras
        R_fixed: Fixed rotation matrix [3, 3] to use for all cameras
        height_z: Fixed Z height for the trajectory
        n_frames: Number of frames to generate
        FoVx, FoVy: Field of view
        width, height: Image dimensions
    
    Returns:
        List of Camera objects following the convex hull perimeter
    """
    # Convert T_all to numpy array and extract XY positions
    T_array = np.array(T_all)  # Shape: [N, 3]
    xy_positions = T_array[:, :2]  # Shape: [N, 2]
    
    # Filter points to 5%~95% range along both x and y axes
    x_min, x_max = np.percentile(xy_positions[:, 0], [5, 95])
    y_min, y_max = np.percentile(xy_positions[:, 1], [5, 95])
    
    mask = (
        (xy_positions[:, 0] >= x_min) & (xy_positions[:, 0] <= x_max) &
        (xy_positions[:, 1] >= y_min) & (xy_positions[:, 1] <= y_max)
    )
    
    xy_positions_filtered = xy_positions[mask]
    
    utils.print_rank_0(f"\nConvex hull filtering:")
    utils.print_rank_0(f"  Original points: {len(xy_positions)}")
    utils.print_rank_0(f"  Filtered points (5%-95% range): {len(xy_positions_filtered)}")
    utils.print_rank_0(f"  X range: [{x_min:.3f}, {x_max:.3f}]")
    utils.print_rank_0(f"  Y range: [{y_min:.3f}, {y_max:.3f}]")
    
    # Compute convex hull on filtered points
    hull = ConvexHull(xy_positions_filtered)
    hull_vertices = hull.vertices  # Indices of points forming the convex hull
    
    # Get the hull points in order (forming a closed loop)
    hull_points = xy_positions_filtered[hull_vertices]
    
    # Close the loop by adding the first point at the end
    hull_points = np.vstack([hull_points, hull_points[0:1]])
    
    utils.print_rank_0(f"\nConvex hull trajectory:")
    utils.print_rank_0(f"  Number of hull vertices: {len(hull_vertices)}")
    utils.print_rank_0(f"  Fixed height (Z): {height_z}")
    utils.print_rank_0(f"  Generating {n_frames} frames")
    
    # Compute cumulative distances along the hull perimeter
    distances = [0.0]
    for i in range(len(hull_points) - 1):
        segment_length = np.linalg.norm(hull_points[i+1] - hull_points[i])
        distances.append(distances[-1] + segment_length)
    
    total_distance = distances[-1]
    
    # Interpolate positions along the hull
    trajectory_positions = []
    for frame_idx in range(n_frames):
        # Compute target distance along the perimeter
        target_distance = (frame_idx / n_frames) * total_distance
        
        # Find which segment this distance falls on
        for i in range(len(distances) - 1):
            if distances[i] <= target_distance <= distances[i+1]:
                # Interpolate within this segment
                segment_start = hull_points[i]
                segment_end = hull_points[i+1]
                segment_length = distances[i+1] - distances[i]
                
                if segment_length > 0:
                    alpha = (target_distance - distances[i]) / segment_length
                else:
                    alpha = 0.0
                
                xy_pos = (1 - alpha) * segment_start + alpha * segment_end
                position_3d = np.array([xy_pos[0], xy_pos[1], height_z])
                trajectory_positions.append(position_3d)
                break
    
    # Create Camera objects for each position
    trajectory_cameras = []
    for idx, T_pos in enumerate(trajectory_positions):
        camera = Camera(
            colmap_id=idx,
            R=R_fixed,
            T=T_pos,
            FoVx=FoVx,
            FoVy=FoVy,
            image=torch.zeros((3, height, width), dtype=torch.float32),
            gt_alpha_mask=None,
            image_name=f"convex_hull_frame_{idx:05d}",
            uid=idx,
            offload=True,
        )
        Camera.original_image_backup = None
        trajectory_cameras.append(camera)
    
    utils.print_rank_0(f"  Generated {len(trajectory_cameras)} cameras along convex hull")
    
    return trajectory_cameras

def generate_convex_hull_trajectory_v2(
    T_all: list,
    R_fixed: np.ndarray,
    height_z: float,
    n_frames: int,
    FoVx: float,
    FoVy: float,
    width: int,
    height: int,
):
    """
    Generate camera trajectory that follows the convex hull of camera positions.
    
    Args:
        T_all: List of all translation vectors from training cameras
        R_fixed: Fixed rotation matrix [3, 3] to use for all cameras
        height_z: Fixed Z height for the trajectory
        n_frames: Number of frames to generate
        FoVx, FoVy: Field of view
        width, height: Image dimensions
    
    Returns:
        List of Camera objects following the convex hull perimeter
    """
    # Convert T_all to numpy array and extract XY positions
    T_array = np.array(T_all)  # Shape: [N, 3]
    xy_positions = T_array[:, :2]  # Shape: [N, 2]
    
    # Filter points to 5%~95% range along both x and y axes
    x_min, x_max = np.percentile(xy_positions[:, 0], [5, 95])
    y_min, y_max = np.percentile(xy_positions[:, 1], [5, 95])
    
    mask = (
        (xy_positions[:, 0] >= x_min) & (xy_positions[:, 0] <= x_max) &
        (xy_positions[:, 1] >= y_min) & (xy_positions[:, 1] <= y_max)
    )
    
    xy_positions_filtered = xy_positions[mask]
    
    utils.print_rank_0(f"\nConvex hull filtering:")
    utils.print_rank_0(f"  Original points: {len(xy_positions)}")
    utils.print_rank_0(f"  Filtered points (5%-95% range): {len(xy_positions_filtered)}")
    utils.print_rank_0(f"  X range: [{x_min:.3f}, {x_max:.3f}]")
    utils.print_rank_0(f"  Y range: [{y_min:.3f}, {y_max:.3f}]")
    
    # Manually provide the hull vertices in original data coordinates
    hull_points = np.array([ # (cam-to-world transformation) world coordinates
        [-20, 20, height_z],
        [0, 35, height_z],
        [10, 30, height_z],
        [15, 20, height_z],
        [12, 0, height_z],
        [-7, 0, height_z],
        [-20, 20, height_z],  # Close the loop
    ]) # this is in world coordinates
    # first number if X, second number is Y, third number is Z in pointcloud_projection.png

    # hull_points = np.array([ # camera in world coordinates
    #     [15, 20, height_z / 2],
    #     [15, 20, height_z],
    #     [15, 20, height_z * 2],
    # ])
    
    utils.print_rank_0(f"\nManual hull vertices:")
    for i, point in enumerate(hull_points[:-1]):  # Skip the last duplicate point
        utils.print_rank_0(f"  Vertex {i}: [{point[0]:.3f}, {point[1]:.3f}]")
    
    # Compute cumulative distances along the hull perimeter
    distances = [0.0]
    for i in range(len(hull_points) - 1):
        segment_length = np.linalg.norm(hull_points[i+1] - hull_points[i])
        distances.append(distances[-1] + segment_length)
    
    total_distance = distances[-1]
    
    # Interpolate positions along the hull
    trajectory_positions = []
    for frame_idx in range(n_frames):
        # Compute target distance along the perimeter
        target_distance = (frame_idx / n_frames) * total_distance
        
        # Find which segment this distance falls on
        for i in range(len(distances) - 1):
            if distances[i] <= target_distance <= distances[i+1]:
                # Interpolate within this segment
                segment_start = hull_points[i]
                segment_end = hull_points[i+1]
                segment_length = distances[i+1] - distances[i]
                
                if segment_length > 0:
                    alpha = (target_distance - distances[i]) / segment_length
                else:
                    alpha = 0.0
                
                camera_pos_in_world = (1 - alpha) * segment_start + alpha * segment_end
                # camera_pos_in_world == cam_to_world.T
                # This is in world coordinates (camera position in world frame)
                # NOTE: R_fixed is world-to-camera rotation matrix (stored transposed)
                R_world_to_camera = R_fixed.T

                # Convert to camera coordinates using: T = -R.T @ camera_position_world
                # where R is camera-to-world rotation (from Camera.update() method)
                world_to_camera_T = - R_world_to_camera @ camera_pos_in_world
                # pos_in_camera = -R_fixed @ pos_in_world # (world-to-camera transformation)

                trajectory_positions.append(world_to_camera_T)
                break
    
    # Create Camera objects for each position
    trajectory_cameras = []
    for idx, T_pos in enumerate(trajectory_positions):
        camera = Camera(
            colmap_id=idx,
            R=R_fixed,
            T=T_pos,
            FoVx=FoVx,
            FoVy=FoVy,
            image=torch.zeros((3, height, width), dtype=torch.float32),
            gt_alpha_mask=None,
            image_name=f"convex_hull_frame_{idx:05d}",
            uid=idx,
            offload=True,
        )
        # print(f"camera.T: {camera.T}")
        # print(f"camera.R: {camera.R}")
        # print(f"camera.camtoworlds: {camera.camtoworlds}")
        # print(f"camera.world_view_transform: {camera.world_view_transform}")
        # print(f"camera.full_proj_transform: {camera.full_proj_transform}")
        # print(f"camera.camera_center: {camera.camera_center}")
        # import pdb; pdb.set_trace()
        Camera.original_image_backup = None
        trajectory_cameras.append(camera)
    
    utils.print_rank_0(f"  Generated {len(trajectory_cameras)} cameras along convex hull")
    
    return trajectory_cameras


def visualize_points_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    sample_rate: float = 0.01,
):
    """
    Visualize point cloud points that are in outlier regions (0-5% or 95-100% along x or y axis).
    Percentiles are computed from the point cloud positions themselves.
    
    Args:
        points: Point cloud positions [N, 3] (xyz)
        colors: Point cloud colors [N, 3] (RGB, range 0-1 or 0-255)
        output_path: Path to save the visualization image
        sample_rate: Fraction of points to sample for visualization (default 0.01 = 1%)
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # Normalize colors to 0-1 range if needed
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Extract XY positions from point cloud
    point_xy = points[:, :2]  # Shape: [N, 2]
    
    # Compute percentiles based on point cloud positions
    x_min, x_max = np.percentile(point_xy[:, 0], [1, 99])
    y_min, y_max = np.percentile(point_xy[:, 1], [1, 99])
    
    # Create masks for outlier points (outside 5%-95% range on either axis)
    # outlier_mask = (
    #     (point_xy[:, 0] < x_min) | (point_xy[:, 0] > x_max) |
    #     (point_xy[:, 1] < y_min) | (point_xy[:, 1] > y_max)
    # )

    outlier_mask = (colors[:, 0] < colors[:, 2]) & (colors[:, 1] < colors[:, 2])
    
    outlier_points = points[outlier_mask]
    outlier_colors = colors[outlier_mask]
    
    # Sample points for visualization
    n_outliers = len(outlier_points)
    n_sample = max(int(n_outliers * sample_rate), 1000)
    n_sample = min(n_sample, n_outliers)
    
    if n_outliers > 0:
        indices = np.random.choice(n_outliers, n_sample, replace=False)
        sampled_points = outlier_points[indices]
        sampled_colors = outlier_colors[indices]
    else:
        sampled_points = np.array([])
        sampled_colors = np.array([])
    
    print(f"\nPoint cloud outlier statistics:")
    print(f"  Total points: {len(points):,}")
    print(f"  Outlier points (0-5% or 95-100%): {n_outliers:,}")
    print(f"  Sampled for visualization: {n_sample:,}")
    print(f"  X range (5%-95%): [{x_min:.3f}, {x_max:.3f}]")
    print(f"  Y range (5%-95%): [{y_min:.3f}, {y_max:.3f}]")
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    
    # Plot outlier points with their original colors
    if len(sampled_points) > 0:
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1],
                  c=sampled_colors, s=0.1, alpha=0.6, rasterized=True)
    
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(f'Point Cloud Outliers (Top-Down View)\n{n_sample:,} / {n_outliers:,} outlier points',
                fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # import pdb; pdb.set_trace()
    
    print(f"Point cloud outliers visualization saved to: {output_path}")
    
    # Save statistics
    stats = {
        "total_points": int(len(points)),
        "outlier_points": int(n_outliers),
        "sampled_points": int(n_sample),
        "sample_rate": float(sample_rate),
        "percentile_ranges": {
            "x": {"min": float(x_min), "max": float(x_max)},
            "y": {"min": float(y_min), "max": float(y_max)},
        }
    }
    
    # stats_path = output_path.replace('.png', '_stats.json')
    # with open(stats_path, 'w') as f:
    #     json.dump(stats, f, indent=2)
    
    # print(f"Point cloud outliers statistics saved to: {stats_path}")
    return stats


def visualize_point_cloud_projection(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    sample_rate: float = 0.01,
    camera_trajectory: Optional[List] = None,
):
    """
    Visualize point cloud with top-down projection to see rough shape.
    
    Args:
        points: Point cloud positions [N, 3] (xyz)
        colors: Point cloud colors [N, 3] (RGB, range 0-1 or 0-255)
        output_path: Path to save the visualization image
        sample_rate: Fraction of points to sample for visualization (default 0.01 = 1%)
        camera_trajectory: Optional list of Camera objects to draw trajectory path
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # Normalize colors to 0-1 range if needed
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Sample points for faster visualization
    n_points = len(points)
    n_sample = max(int(n_points * sample_rate), 1000)  # At least 1000 points
    n_sample = min(n_sample, n_points)  # Don't exceed total points
    
    indices = np.random.choice(n_points, n_sample, replace=False)
    sampled_points = points[indices]
    sampled_colors = colors[indices]
    
    print(f"Visualizing {n_sample:,} / {n_points:,} points ({sample_rate*100:.1f}%)")
    
    # Create figure with single top-down view (higher resolution)
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    
    # Top-down view (XY projection, looking down Z-axis)
    # Use smaller point size (0.1 instead of 1) and slightly higher alpha
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                c=sampled_colors, s=0.1, alpha=0.6, rasterized=True)
    
    # Draw camera trajectory if provided
    if camera_trajectory is not None and len(camera_trajectory) > 0:
        # Extract camera positions (XY coordinates)
        # cam_positions = [
        #     [-20, 20],
        #     [0, 35],
        #     [10, 30],
        #     [15, 20],
        #     [12, 0],
        #     [-7, 0],
        #     [-20, 20],
        # ]
        cam_positions = []
        for cam in camera_trajectory:
            # Camera position is -R^T @ T
            if hasattr(cam, 'T') and hasattr(cam, 'R'):
                R = cam.R if isinstance(cam.R, np.ndarray) else cam.R
                T = cam.T if isinstance(cam.T, np.ndarray) else cam.T # R and T are world-to-camera

                # R == world-to-camera rotation matrix (stored transposed)
                R_world_to_camera = R.T
                T_world_to_camera = T
                # Camera center in world coordinates
                cam_center = - R_world_to_camera.T @ T_world_to_camera
                # import pdb; pdb.set_trace()
                cam_positions.append([cam_center[0], cam_center[1]])
                # print(f"cam_center: {cam_center}")
                # print(f"cam.T: {cam.T}")
                # print(f"cam.R: {cam.R}")
                # print(f"cam.camtoworlds: {cam.camtoworlds}")
                # print(f"cam.world_view_transform: {cam.world_view_transform}")
                # print(f"cam.full_proj_transform: {cam.full_proj_transform}")
                # print(f"cam.camera_center: {cam.camera_center}")
                # import pdb; pdb.set_trace()

        
        if len(cam_positions) > 0:
            cam_positions = np.array(cam_positions)
            # Draw trajectory path
            ax.plot(cam_positions[:, 0], cam_positions[:, 1], 
                   'r-', linewidth=2, alpha=0.8, label='Camera Trajectory')
            # Draw start and end points
            ax.scatter(cam_positions[0, 0], cam_positions[0, 1], 
                      c='green', s=100, marker='o', edgecolors='black', 
                      linewidth=2, label='Start', zorder=10)
            ax.scatter(cam_positions[-1, 0], cam_positions[-1, 1], 
                      c='red', s=100, marker='s', edgecolors='black', 
                      linewidth=2, label='End', zorder=10)
            # Add arrow to show direction
            if len(cam_positions) > 1:
                # Add arrows at intervals along the trajectory
                arrow_interval = max(len(cam_positions) // 10, 1)
                for i in range(0, len(cam_positions) - 1, arrow_interval):
                    dx = cam_positions[i+1, 0] - cam_positions[i, 0]
                    dy = cam_positions[i+1, 1] - cam_positions[i, 1]
                    ax.arrow(cam_positions[i, 0], cam_positions[i, 1], 
                            dx * 0.3, dy * 0.3,
                            head_width=0.5, head_length=0.3, 
                            fc='red', ec='red', alpha=0.6, zorder=9)
            ax.legend(loc='upper right', fontsize=12)
    
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    
    title = f'Top-Down View (XY Projection)\n{n_sample:,} / {n_points:,} points'
    if camera_trajectory is not None and len(camera_trajectory) > 0:
        title += f' | {len(camera_trajectory)} camera positions'
    ax.set_title(title, fontsize=16)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure with higher DPI (300 instead of 150)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Point cloud visualization saved to: {output_path}")
    
    # Also create a simple statistics summary
    stats = {
        "total_points": int(n_points),
        "sampled_points": int(n_sample),
        "sample_rate": float(sample_rate),
        "bounds": {
            "x": {"min": float(points[:, 0].min()), "max": float(points[:, 0].max())},
            "y": {"min": float(points[:, 1].min()), "max": float(points[:, 1].max())},
            "z": {"min": float(points[:, 2].min()), "max": float(points[:, 2].max())},
        }
    }
    
    if camera_trajectory is not None:
        stats["camera_trajectory_frames"] = len(camera_trajectory)
    
    stats_path = output_path.replace('.png', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Point cloud statistics saved to: {stats_path}")
    return stats


def render_single_image(
    camera: Camera,
    gaussians,
    background: torch.Tensor,
    save_path: str,
    args,
    scene: Scene = None,
):
    """
    Render a single image from a camera viewpoint and save it to disk.
    
    Args:
        camera: Camera object defining the viewpoint
        gaussians: Gaussian model containing the 3D scene representation
        background: Background color tensor
        save_path: Path to save the rendered image (e.g., 'output/image_001.png')
        args: Command-line arguments containing offload strategy flags
        scene: Scene object (optional, used for some offload strategies)
    
    Returns:
        rendered_image: The rendered image tensor [H, W, 3]
    """
    # Prepare camera matrices on GPU
    camera.world_view_transform = camera.world_view_transform.cuda()
    camera.full_proj_transform = camera.full_proj_transform.cuda()
    camera.K = camera.create_k_on_gpu()
    camera.camtoworlds = torch.inverse(
        camera.world_view_transform.transpose(0, 1)
    ).unsqueeze(0)
    # print(f"camera.camtoworlds: {camera.camtoworlds}")
    print("camera.world_view_transform: ", camera.world_view_transform)
    
    # Render frame based on offload strategy
    if args.naive_offload:
        rendered_image = naive_offload_eval_one_cam(
            gaussians=gaussians,
            scene=scene,
            camera=camera,
            background=background,
        )
    elif args.clm_offload:
        rendered_image = clm_offload_eval_one_cam(
            camera=camera,
            gaussians=gaussians,
            background=background,
            scene=scene,
        )
    elif args.no_offload:
        rendered_image, _, _, _ = baseline_accumGrads_micro_step(
            means3D=gaussians.get_xyz,
            opacities=gaussians.get_opacity,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            shs=gaussians.get_features,
            sh_degree=gaussians.active_sh_degree,
            camera=camera,
            background=background,
            mode="eval",
        )
    else:
        raise ValueError("Invalid offload configuration")
    
    # Process rendered output
    colors = torch.clamp(rendered_image, 0.0, 1.0)  # [H, W, 3] or [3, H, W]
    
    # Ensure colors are in [H, W, 3] format
    if colors.shape[0] == 3:
        colors = colors.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    
    # Convert to numpy for image saving
    colors_np = (colors * 255).cpu().to(torch.uint8).numpy()
    
    # Save image to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not args.save_video:
        imageio.imwrite(save_path, colors_np)
    
    # Clean up GPU memory
    camera.world_view_transform = camera.world_view_transform.cpu()
    camera.full_proj_transform = camera.full_proj_transform.cpu()
    torch.cuda.empty_cache()
    
    return colors

def create_rotation_matrix(degree: float, axis: str):
    """
    Create a rotation matrix for a given degree around a given axis.
    """
    if axis == "x":
        return np.array([[1, 0, 0], [0, np.cos(degree), -np.sin(degree)], [0, np.sin(degree), np.cos(degree)]])
    elif axis == "y":
        return np.array([[np.cos(degree), 0, np.sin(degree)], [0, 1, 0], [-np.sin(degree), 0, np.cos(degree)]])
    elif axis == "z":
        return np.array([[np.cos(degree), -np.sin(degree), 0], [np.sin(degree), np.cos(degree), 0], [0, 0, 1]])

def main():
    """Main function for trajectory rendering."""
    # Parse command-line arguments
    parser = ArgumentParser(description="Trajectory rendering script")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    
    # Add trajectory-specific arguments
    parser.add_argument("--traj_path", type=str, default="original",
                    #    choices=["interp", "ellipse", "spiral", "original", "manual"],
                       help="Type of camera trajectory")
    parser.add_argument("--manual_height", type=float, default=30.0,
                       help="Fixed Z height for manual convex hull trajectory")
    parser.add_argument("--n_frames", type=int, default=120,
                       help="Number of frames to generate (ignored for 'original')")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/render_images)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (default: -1 = load latest saved iteration)")
    parser.add_argument("--save_video", action="store_true",
                       help="Also save rendered images as a video")
    parser.add_argument("--visualize_pointcloud", action="store_true",
                       help="Create vertical projection visualization of point cloud")
    parser.add_argument("--pointcloud_sample_rate", type=float, default=0.01,
                       help="Sampling rate for point cloud visualization (default: 0.01 = 1%%)")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize arguments
    init_args(args)
    args = utils.get_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "render_images")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = open(os.path.join(args.output_dir, "render_images.log"), "w")
    utils.set_log_file(log_file)
    
    utils.print_rank_0("=" * 80)
    utils.print_rank_0("Image Rendering")
    utils.print_rank_0("=" * 80)
    utils.print_rank_0(f"Model path: {args.model_path}")
    utils.print_rank_0(f"Load iteration: {args.iteration} (-1 = latest)")
    utils.print_rank_0(f"Trajectory type: {args.traj_path}")
    utils.print_rank_0(f"Number of frames: {args.n_frames}")
    utils.print_rank_0(f"Output directory: {args.output_dir}")
    utils.print_rank_0(f"Save video: {args.save_video}")
    utils.print_rank_0("=" * 80)
    
    # Initialize Gaussian model based on offload strategy
    dataset_args = lp.extract(args)
    
    if args.naive_offload:
        gaussians = GaussianModelNaiveOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNaiveOffload")
    elif args.clm_offload:
        gaussians = GaussianModelCLMOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelCLMOffload")
    elif args.no_offload:
        gaussians = GaussianModelNoOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNoOffload (no offload, GPU-only)")
    else:
        raise ValueError(
            f"Invalid offload configuration: naive_offload={args.naive_offload}, "
            f"clm_offload={args.clm_offload}, no_offload={args.no_offload}"
        )
    
    # Load scene info and trained model
    # Note: We load camera metadata WITHOUT decoding images to disk (rendering-only workflow)
    # The PLY files are saved at: model_path/point_cloud/iteration_{iteration}/point_cloud.ply
    with torch.no_grad():
        utils.print_rank_0(f"\nLoading scene info and model from iteration {args.iteration}...")
        
        # Load scene information (camera poses, intrinsics) without image decoding
        scene_info, cameras_extent = load_scene_info_for_rendering(args)
        utils.print_rank_0("Loaded scene metadata (camera poses and intrinsics)")
        # import pdb; pdb.set_trace()
        
        if args.load_pt_path != '':
            gaussians.load_tensors(args.load_pt_path)
            utils.print_rank_0(f"Loaded Gaussians from {args.load_pt_path}")
        elif args.load_ply_path != '':
            gaussians.load_ply(args.load_ply_path)
            utils.print_rank_0(f"Loaded Gaussians from {args.load_ply_path}")
        else:
            if args.iteration == -1:
                loaded_iter = searchForMaxIteration(
                    os.path.join(args.model_path, "point_cloud")
                )
            else:
                loaded_iter = args.iteration
            ply_path = os.path.join(
                args.model_path, "point_cloud", f"iteration_{loaded_iter}"
            )
            gaussians.load_ply(ply_path)
            utils.print_rank_0(f"Loaded Gaussians from iteration {loaded_iter}...") 

        utils.print_rank_0(f"Loaded {len(gaussians._xyz)} Gaussians.")
        
        # Set active SH degree to maximum for best quality rendering
        # During training, SH degree is progressively increased
        # For rendering, we use the maximum degree for best visual quality
        gaussians.active_sh_degree = gaussians.max_sh_degree
    
    # import pdb; pdb.set_trace()

    # know the image width and height
    image_width = scene_info.train_cameras[0].width
    image_height = scene_info.train_cameras[0].height

    # utils.get_img_width()
    utils.set_img_size(image_height, image_width)

    # Setup background
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    world_to_camera_all = []
    camtoworlds_all = []
    T_all = []  # Collect all translation vectors
    # K_all = []
    def getWorld2View(R, t):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        # cam_center = C2W[:3, 3]
        # cam_center = (cam_center + translate) * scale
        # C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    # def getK(FovX, FovY, image_width, image_height):
    #     tanfovx = math.tan(FovX * 0.5)
    #     tanfovy = math.tan(FovY * 0.5)
    #     focal_length_x = image_width / (2 * tanfovx)
    #     focal_length_y = image_height / (2 * tanfovy)
    #     K = torch.tensor(
    #         [
    #             [focal_length_x, 0, image_width / 2.0],
    #             [0, focal_length_y, image_height / 2.0],
    #             [0, 0, 1],
    #         ],
    #         # device="cuda",
    #     )
    #     return K

    camera_positions_all = []  # Store camera positions in world space
    for id, cam_info in tqdm(
        enumerate(scene_info.train_cameras), total=len(scene_info.train_cameras)
    ):
        world_view_transform = (
            torch.tensor(getWorld2View(cam_info.R, cam_info.T)).transpose(0, 1)
        )
        world_to_camera_all.append(world_view_transform)
        view_world_transform = world_view_transform.t().inverse()
        camtoworlds_all.append(view_world_transform)
        T_all.append(cam_info.T)  # Collect translation vector
        # Extract camera position from view-to-world transformation
        camera_position = view_world_transform[:3, 3].cpu().numpy()
        camera_positions_all.append(camera_position)
        # K_all.append(getK(cam_info.FovX, cam_info.FovY, image_width, image_height))
    
    # Print statistics for camera positions (view-to-world) in JSON format
    if camera_positions_all:
        camera_pos_array = np.array(camera_positions_all)  # Shape: [N, 3]
        stats = {
            "camera_position_statistics": {
                "X": {
                    "min": float(camera_pos_array[:, 0].min()),
                    "max": float(camera_pos_array[:, 0].max())
                },
                "Y": {
                    "min": float(camera_pos_array[:, 1].min()),
                    "max": float(camera_pos_array[:, 1].max())
                },
                "Z": {
                    "min": float(camera_pos_array[:, 2].min()),
                    "max": float(camera_pos_array[:, 2].max())
                }
            }
        }
        utils.print_rank_0("\nCamera Position (View-to-World) Statistics (JSON):")
        utils.print_rank_0(json.dumps(stats, indent=2))
    
    
    FoVx = scene_info.train_cameras[0].FovX
    FoVy = scene_info.train_cameras[0].FovY

    # Generate trajectory cameras
    if args.traj_path == "original":
        trajectory_cameras = []
        for idx, cam_info in enumerate(scene_info.train_cameras[:args.n_frames]):
            # import pdb; pdb.set_trace()
            cam_info.T[2] = 30 # set the z-axis (height) to 30
            camera = create_camera_from_cam_info(
                cam_info=cam_info,
                uid=idx,
            )
            trajectory_cameras.append(camera)
    elif args.traj_path == "manual":
        # Use convex hull trajectory with fixed rotation and height
        R_fixed = scene_info.train_cameras[0].R
        
        # # Visualize point cloud outliers before generating trajectory
        # outlier_viz_path = os.path.join(args.output_dir, "pointcloud_outliers.png")
        # visualize_points_outliers(
        #     points=scene_info.point_cloud.points,
        #     colors=scene_info.point_cloud.colors,
        #     output_path=outlier_viz_path,
        #     sample_rate=args.pointcloud_sample_rate,
        # )
        
        trajectory_cameras = generate_convex_hull_trajectory(
            T_all=T_all,
            R_fixed=R_fixed,
            height_z=args.manual_height,
            n_frames=args.n_frames,
            FoVx=FoVx,
            FoVy=FoVy,
            width=image_width,
            height=image_height,
        )
    elif args.traj_path == "manual_v2":
        # Use convex hull trajectory with fixed rotation and height
        R_fixed = scene_info.train_cameras[0].R
        # R_fixed is world-to-camera rotation matrix (stored transposed) = camera-to-world rotation matrix

        # Rotate R_fixed around x-axis
        angle = np.radians(30)  # Rotation angle in degrees (adjust as needed)
        new_R = create_rotation_matrix(angle, "y")
        R_fixed = new_R @ R_fixed # This performs reasonably well. 

        # R_fixed is a [[1,0,0], [0,1,0], [0,0,-1]] matrix

        R_fixed = np.array([[1,0,0], [0,1,0], [0,0,-1]]) # this scene is easy that we can use this simple rotation matrix. 
        
        # # Visualize point cloud outliers before generating trajectory
        # outlier_viz_path = os.path.join(args.output_dir, "pointcloud_outliers.png")
        # visualize_points_outliers(
        #     points=scene_info.point_cloud.points,
        #     colors=scene_info.point_cloud.colors,
        #     output_path=outlier_viz_path,
        #     sample_rate=args.pointcloud_sample_rate,
        # )
        
        trajectory_cameras = generate_convex_hull_trajectory_v2(
            T_all=T_all,
            R_fixed=R_fixed,
            height_z=args.manual_height,
            n_frames=args.n_frames,
            FoVx=FoVx,
            FoVy=FoVy,
            width=image_width,
            height=image_height,
        )

    else:
        trajectory_cameras = generate_trajectory_cameras(
            world_to_camera_all=world_to_camera_all,
            camtoworlds_all=camtoworlds_all,
            traj_path=args.traj_path,
            n_frames=args.n_frames,
            FoVx=FoVx,
            FoVy=FoVy,
            width=image_width,
            height=image_height,
        )

    # Visualize point cloud with trajectory if requested
    if args.visualize_pointcloud:
        utils.print_rank_0("\n" + "=" * 80)
        utils.print_rank_0("Creating Point Cloud Visualization with Camera Trajectory")
        utils.print_rank_0("=" * 80)
        
        # Get point cloud data from scene_info
        points = scene_info.point_cloud.points
        colors = scene_info.point_cloud.colors
        
        # Create visualization with trajectory
        viz_path = os.path.join(args.output_dir, "pointcloud_projection.png")
        visualize_point_cloud_projection(
            points=points,
            colors=colors,
            output_path=viz_path,
            sample_rate=args.pointcloud_sample_rate,
            camera_trajectory=trajectory_cameras,
        )
        utils.print_rank_0("=" * 80)

    # import pdb; pdb.set_trace()

    # Render cameras directly
    total_frames = len(trajectory_cameras)
    utils.print_rank_0(f"\nRendering {total_frames} images...")
    
    # Optional video writer
    writer = None
    if args.save_video:
        video_path = os.path.join(args.output_dir, f"trajectory_{args.traj_path}.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        utils.print_rank_0(f"Video will be saved to: {video_path}")
    
    # Render each camera
    for frame_idx in tqdm(range(total_frames), desc="Rendering images"):
        camera = trajectory_cameras[frame_idx]
        
        # Generate save path for this frame
        save_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}.png")
        
        # Render and save single image
        rendered_colors = render_single_image(
            camera=camera,
            gaussians=gaussians,
            background=background,
            save_path=save_path,
            args=args,
            scene=None,
        )
        
        # Optionally add to video
        if args.save_video:
            colors_np = (rendered_colors * 255).cpu().to(torch.uint8).numpy()
            writer.append_data(colors_np)
        
        # Clean up
        del rendered_colors
        torch.cuda.empty_cache()
    
    # Close video writer if used
    if args.save_video:
        writer.close()
        utils.print_rank_0(f"Video saved to: {video_path}")
    
    # Cleanup
    log_file.write(f"Rendering completed. Images saved to: {args.output_dir}\n")
    if args.save_video:
        log_file.write(f"Video saved to: {video_path}\n")
    log_file.close()
    utils.print_rank_0("\nRendering complete!")
    utils.print_rank_0(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
