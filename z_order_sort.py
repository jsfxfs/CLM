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

import os
import torch
import sys
from utils.general_utils import (
    safe_state,
    prepare_output_and_logger,
)
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
import utils.general_utils as utils
from utils.general_utils import safe_state, init_distributed
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    init_args,
)
import numpy as np
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.timer import Timer
import math
from gsplat import fully_fused_projection
import random
import json
import zordersort
from tqdm import tqdm

def save_ply_file(file_path, points3d):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    elements = np.empty(len(points3d), dtype=dtype)
    elements[:] = list(map(tuple, points3d))
    vertex = PlyElement.describe(elements, 'vertex')
    PlyData([vertex]).write(file_path)

def get_all_gaussians(pcd: BasicPointCloud):
    args = utils.get_args()

    if hasattr(args, "load_ply_path") and args.load_ply_path != "":
        folder = args.load_ply_path

        world_size = -1
        for f in os.listdir(folder):
            if "_ws" in f:
                world_size = int(f.split("_ws")[1].split(".")[0])
                break
        assert world_size > 0, "world_size should be greater than 1."

        def load_raw_ply(path):
            print("Loading ", path)
            plydata = PlyData.read(path)

            xyz = np.stack(
                (
                    np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"]),
                ),
                axis=1,
            )
            scale_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("scale_")
            ]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            rot_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
            ]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            return xyz, scales, rots

        catted_xyz = []
        catted_scaling = []
        catted_rotation = []
        for rk in range(world_size):
            one_checkpoint_path = (
                folder + "/point_cloud_rk" + str(rk) + "_ws" + str(world_size) + ".ply"
            )
            xyz, scales, rots = (
                load_raw_ply(one_checkpoint_path)
            )
            catted_xyz.append(xyz)
            catted_scaling.append(scales)
            catted_rotation.append(rots)
        catted_xyz = np.concatenate(catted_xyz, axis=0)
        catted_scaling = np.concatenate(catted_scaling, axis=0)
        catted_rotation = np.concatenate(catted_rotation, axis=0)

        fused_point_cloud = (
            torch.tensor(catted_xyz).float().cuda().contiguous()
        )
        scales = torch.tensor(catted_scaling).float().cuda()
        rots = torch.tensor(catted_rotation).float().cuda()

        print("fused_point_cloud.shape: ", fused_point_cloud.shape)
        print("scales.shape: ", scales.shape)
        print("rots.shape: ", rots.shape)
        return fused_point_cloud, scales, rots


    fused_point_cloud = (
        torch.tensor(np.asarray(pcd.points)).float().cuda()
    )  # It is not contiguous
    fused_point_cloud = fused_point_cloud.contiguous()  # Now it's contiguous

    print(
        "Number of points before initialization : ", fused_point_cloud.shape[0]
    )

    dist2 = torch.clamp_min(
        distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
        0.0000001,
    )
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    return fused_point_cloud, scales, rots

def visibility_gaussian_ids(camera, fused_point_cloud, scales, rots):
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    focal_length_x = camera.image_width / (2 * tanfovx)
    focal_length_y = camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, camera.image_width / 2.0],
            [0, focal_length_y, camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    viewmat = camera.world_view_transform.transpose(0, 1)

    proj_results = fully_fused_projection(
        means=fused_point_cloud,  # (N, 3)
        covars=None,
        quats=torch.nn.functional.normalize(rots),
        scales=torch.exp(scales),
        # quats=rots,
        # scales=scales,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=camera.image_width,
        height=camera.image_height,
        packed=True, # use packed mode here
        # near_plane = 0.01,
        # far_plane = 1e10,
        # radius_clip = args.radius_clip, # use default values for now
        sparse_grad=False,
    )

    # The results are packed into shape [nnz, ...]. All elements are valid.
    (
        camera_ids, # (nnz,)
        gaussian_ids, # (nnz,)
        radii_packed, # (nnz,)
        means2d_packed, # (nnz, 2)
        depths_packed, # (nnz,)
        conics_packed, # (nnz, 3)
        _,# compensations
        # indptr, # (num_cameras + 1, )
    ) = proj_results

    return gaussian_ids

def check_dataset(args):
    utils.log_cpu_memory_usage("before loading images meta data")

    if os.path.exists(
        os.path.join(args.source_path, "sparse")
    ):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.images, args.eval, args.llffhold
        )
    elif "matrixcity" in args.source_path:  # This is for matrixcity
        scene_info = sceneLoadTypeCallbacks["City"](
            args.source_path,
            args.random_background,
            args.white_background,
            llffhold=args.llffhold,
        )
    else:
        raise ValueError("No valid dataset found in the source path")

    with open(scene_info.ply_path, "rb") as src_file, open(
        os.path.join(args.model_path, "input.ply"), "wb"
    ) as dest_file:
        dest_file.write(src_file.read())

    utils.log_cpu_memory_usage("before decoding images")

    cameras_extent = scene_info.nerf_normalization["radius"]

    # Set image size to global variable
    orig_w, orig_h = (
        scene_info.train_cameras[0].width,
        scene_info.train_cameras[0].height,
    )
    utils.set_img_size(orig_h, orig_w)

    # output the number of cameras in the training set and image size to the log file
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
    fused_point_cloud, scales, rots = get_all_gaussians(scene_info.point_cloud) # (N, 3), (N, 3), (N, 4) in cuda

    if fused_point_cloud.shape[0] > 5000000:
        random_indices = torch.randperm(fused_point_cloud.shape[0])[:5000000].cuda()
        fused_point_cloud = fused_point_cloud[random_indices]
        scales = scales[random_indices]
        rots = rots[random_indices]

    # z-order sort
    fused_point_cloud_np = fused_point_cloud.cpu().numpy().astype(np.float32).copy()
    fused_point_cloud_np = fused_point_cloud_np[:, :2].copy()
    sorted_fused_point_cloud_np = fused_point_cloud_np.copy()
    rank2id = np.zeros(fused_point_cloud_np.shape[0], dtype=np.uint32)
    zordersort.zordersort(fused_point_cloud_np, sorted_fused_point_cloud_np, rank2id, 2)
    rank2id_torch = torch.from_numpy(rank2id.astype(np.int32)).cuda()

    # sort the fused_point_cloud, scales, rots
    sorted_fused_point_cloud = fused_point_cloud[rank2id_torch]
    sorted_scales = scales[rank2id_torch]
    sorted_rots = rots[rank2id_torch]

    # n_gaussians = sorted_fused_point_cloud.shape[0]
    # sorted_fused_point_cloud = torch.cat([sorted_fused_point_cloud[n_gaussians//10:], sorted_fused_point_cloud[:n_gaussians//10]], dim=0)
    # sorted_scales = torch.cat([sorted_scales[n_gaussians//10:], sorted_scales[:n_gaussians//10]], dim=0)
    # sorted_rots = torch.cat([sorted_rots[n_gaussians//10:], sorted_rots[:n_gaussians//10]], dim=0)


    # this is for debug
    # sorted_fused_point_cloud = fused_point_cloud
    # sorted_scales = scales
    # sorted_rots = rots

    safe_state(args.quiet)
    train_camera_filters = []
    total_visible_gaussians = 0
    total_total_gaussians = 0
    for camera in train_cameras:
        train_camera_filters.append(visibility_gaussian_ids(camera, sorted_fused_point_cloud, sorted_scales, sorted_rots))
        print("Camera: ", camera.image_name)
        # print("Num of visible Gaussians: ", len(train_camera_filters[-1]))
        print("Percent of visible Gaussians: ", round(1.0*len(train_camera_filters[-1])/sorted_fused_point_cloud.shape[0], 2))
        # print("Min visible Gaussian id: ", torch.min(train_camera_filters[-1]).item())
        # print("Max visible Gaussian id: ", torch.max(train_camera_filters[-1]).item())
        total_visible_gaussians += len(train_camera_filters[-1])
        total_total_gaussians += sorted_fused_point_cloud.shape[0]
    
    print("Total visible Gaussians: ", total_visible_gaussians)
    print("Total total Gaussians: ", total_total_gaussians)
    print("Percent of visible Gaussians: ", round(1.0*total_visible_gaussians/total_total_gaussians, 2))

    # os.makedirs(os.path.join(args.model_path, f"overlap_size_matrix"), exist_ok=True)
    repeat_times = 50
    bsz = 16
    all_results = []
    gaussian_block_size = 128 # contiguous 128 gaussians
    for iter in tqdm(range(repeat_times)):
        sampled_cameras_indices = random.sample(range(len(train_cameras)), bsz)
        sampled_filters = [train_camera_filters[i] for i in sampled_cameras_indices]
        sampled_filters_list = [filter.cpu().numpy().tolist() for filter in sampled_filters]
        overlap_size_matrix = [[0 for _ in range(bsz)] for _ in range(bsz)]

        bool_filters = []
        for filter in sampled_filters:
            bool_filter = torch.zeros(sorted_fused_point_cloud.shape[0], dtype=torch.int).cuda()
            bool_filter[filter] = 1
            bool_filters.append(bool_filter)

        for i in range(bsz):
            for j in range(i+1, bsz):
                overlap_size_matrix[i][j] = (bool_filters[i]*bool_filters[j]).sum().item()
                overlap_size_matrix[j][i] = overlap_size_matrix[i][j]
        # json.dump(overlap_size_matrix, open(os.path.join(args.model_path, f"overlap_size_matrix/{iter}.json"), "w"))

        sum_filters = torch.stack(bool_filters, dim=0).sum(dim=0)
        total_visible_gaussians = sum_filters.sum().item()
        n_finished_gaussians = []
        n_finished_gaussians_blocks = []
        overlap_size = []
        ordered_indices = [0]
        for i in range(1, bsz):
            sum_filters_0 = (sum_filters == 0)
            n_finished_gaussians.append(sum_filters_0.sum().item())
            sum_filters_0_block = sum_filters_0[:sum_filters_0.shape[0] // gaussian_block_size * gaussian_block_size].view(-1, gaussian_block_size)
            sum_filters_0_block = sum_filters_0_block.sum(dim=1)
            sum_filters_0_block_0 = (sum_filters_0_block == gaussian_block_size)
            n_finished_gaussians_blocks.append(sum_filters_0_block_0.sum().item())


            cur_index = ordered_indices[-1]
            sum_filters -= bool_filters[cur_index]

            next_index = -1
            max_overlap = -1
            for j in range(bsz):
                if j in ordered_indices:
                    continue
                overlap = (bool_filters[cur_index] * bool_filters[j]).sum().item()
                if overlap > max_overlap:
                    next_index = j
                    max_overlap = overlap
            ordered_indices.append(next_index)
            overlap_size.append(max_overlap)
        
        result = {
            "total_gaussians": sorted_fused_point_cloud.shape[0],
            "total_visible_gaussians_count": total_visible_gaussians,
            "total_overlap_gaussians_count": sum(overlap_size),
            "overlap_size": str(overlap_size),
            "n_finished_gaussians": str(n_finished_gaussians),
            "n_total_blocks": sorted_fused_point_cloud.shape[0] // gaussian_block_size,
            "n_finished_gaussians_blocks": str(n_finished_gaussians_blocks),
            "overlap_size_matrix": str(overlap_size_matrix),
        }
        all_results.append(result)

    json.dump(all_results, open(os.path.join(args.model_path, "all_results.json"), "w"), indent=4)




        # # print the matrix
        # print("Overlap Size Matrix: ")
        # for row in overlap_size_matrix:
        #     print(row)

        ### Sort the filters
        # sampled_key_for_sort = [
        #     (torch.min(filter).item(), torch.max(filter).item()) for filter in sampled_filters
        # ]
        # indices = list(range(bsz))
        # indices.sort(key=lambda i: sampled_key_for_sort[i])
        # sorted_keys = [sampled_key_for_sort[i] for i in indices]

        # print("Iteration: ", iter)
        # print("Sorted Keys: ", sorted_keys)

    #### Save z-order partition
    # n_partitions = 1000
    # n_points = sorted_fused_point_cloud.shape[0]
    # partition_size = n_points // n_partitions

    # os.makedirs(os.path.join(args.model_path, "partitions"), exist_ok=True)

    # sorted_fused_point_cloud_np = sorted_fused_point_cloud.cpu().numpy()

    # partition_l = 0
    # partition_r = 0
    # for i in range(n_partitions):
    #     partition_l = partition_r
    #     partition_r = partition_l + partition_size + (1 if i < n_points % n_partitions else 0)
    #     save_ply_file(os.path.join(args.model_path, f"partitions/partition{i}.ply"), sorted_fused_point_cloud_np[partition_l:partition_r])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)

    # print("Before init_distributed")

    args = parser.parse_args(sys.argv[1:])

    # Set up distributed training
    init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    # set random seed
    safe_state(args.quiet)

    args = utils.get_args()

    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(args.log_folder + "/python.log", "w")

    utils.set_log_file(log_file)

    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    start_from_this_iteration = 1

    with torch.no_grad():
        check_dataset(args)




# nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o 