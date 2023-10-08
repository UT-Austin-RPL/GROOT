import os

import h5py
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from einops import rearrange

from torch.multiprocessing import Pool, Process, set_start_method, get_start_method

from functools import partial

from vos_3d_algo.o3d_modules import convert_convention, O3DPointCloud
from vos_3d_algo.misc_utils import resize_image_to_same_shape, get_first_frame_annotation, normalize_pcd, rotate_camera_pose, get_transformed_depth_img, depth_to_rgb, VideoWriter, Timer
from vos_3d_algo.point_mae_modules import Group

def scene_pcd_fn(
        cfg,
        rgb_img_input,
        depth_img_input,
        intrinsic_matrix,
        extrinsic_matrix,
        max_points=10000,
        is_real_robot=True,
    ):
        rgbd_pc = O3DPointCloud(max_points=max_points)
        rgbd_pc.create_from_rgbd(rgb_img_input, depth_img_input, intrinsic_matrix)
        rgbd_pc.transform(extrinsic_matrix)
        rgbd_pc.preprocess()

        return rgbd_pc.get_points(), rgbd_pc.get_colors()
    

def object_pcd_fn(
        cfg,
        rgb_img_input,
        depth_img_input,
        mask_img_input,
        intrinsic_matrix,
        extrinsic_matrix,
        first_frame_annotation,
        prev_xyz=None,
        is_real_robot=True,
        erode_boundary=False,
        erod_boundary_kernel_size=3,
        ):
    """function for generation object point clouds

    Args:
        cfg (dict, or Easydict): experiment configuration
        rgb_img_input (np.ndarray): rgb image
        depth_img_input (np.ndarray): depth image
        mask_img_input (np.ndarray): mask image
        intrinsic_matrix (np.ndarray): intrinsic matrix
        extrinsic_matrix (np.ndarray): extrinsic matrix
        first_frame_annotation (np.ndarray): first frame annotation
        prev_xyz (np.ndarray, optional): previous xyz. Defaults to None.
        is_real_robot (bool, optional): whether it's real robot. Defaults to True.
        erode_boundary (bool, optional): whether to erode the boundary. Defaults to False.
        erod_boundary_kernel_size (int, optional): kernel size for erosion. Defaults to 3.
    Return: 
        np.ndarray: object point cloud (NUM_OBJECTS, MAX_NUM, 3)
    """

    rgb_img = convert_convention(rgb_img_input, real_robot=is_real_robot)
    depth_img = convert_convention(depth_img_input, real_robot=is_real_robot)

    mask_img = resize_image_to_same_shape(mask_img_input, rgb_img)
    if erode_boundary:
        kernel = np.ones((erod_boundary_kernel_size, erod_boundary_kernel_size),np.uint8)

    xyz_list = []
    for mask_idx in range(1, first_frame_annotation.max() + 1):
        masked_depth_img = depth_img.copy()
        binary_mask = np.where(mask_img == mask_idx, 1, 0)

        if erode_boundary:
            binary_mask = binary_erosion(binary_mask, kernel)

        masked_depth_img[binary_mask == 0] = -1

        if np.sum(binary_mask) == 0:
            if prev_xyz is None:
                points = np.zeros((cfg.datasets.max_points, 3))
            else:
                points = prev_xyz[mask_idx - 1]
        else:
            o3d_pc = O3DPointCloud(max_points=cfg.datasets.max_points)
            o3d_pc.create_from_depth(masked_depth_img, intrinsic_matrix)
            o3d_pc.transform(extrinsic_matrix)
            if (o3d_pc.get_num_points() == 0):
                raise ValueError("No points in the point cloud")
            o3d_pc.preprocess(use_rgb=False)

            points = o3d_pc.get_points()
        
        xyz_list.append(points)

    return np.stack(xyz_list, axis=0)

def object_pcd_aug_fn(
        cfg,
        rgb_img_input,
        depth_img_input,
        mask_img_input,
        intrinsic_matrix,
        original_extrinsic_matrix,
        new_extrinsic_matrix,
        first_frame_annotation,
        prev_xyz=None,
        is_real_robot=True,
        erode_boundary=False,
        erod_boundary_kernel_size=3,
        ):
    """function for generation object point clouds

    Args:
        cfg (dict, or Easydict): experiment configuration
        rgb_img_input (np.ndarray): rgb image
        depth_img_input (np.ndarray): depth image
        mask_img_input (np.ndarray): mask image
        intrinsic_matrix (np.ndarray): intrinsic matrix
        extrinsic_matrix (np.ndarray): extrinsic matrix
        first_frame_annotation (np.ndarray): first frame annotation
        prev_xyz (np.ndarray, optional): previous xyz. Defaults to None.
        is_real_robot (bool, optional): whether it's real robot. Defaults to True.
        erode_boundary (bool, optional): whether to erode the boundary. Defaults to False.
        erod_boundary_kernel_size (int, optional): kernel size for erosion. Defaults to 3.
    Return: 
        np.ndarray: object point cloud (NUM_OBJECTS, MAX_NUM, 3)
    """

    rgb_img = convert_convention(rgb_img_input, real_robot=is_real_robot)
    depth_img = convert_convention(depth_img_input, real_robot=is_real_robot)

    mask_img = resize_image_to_same_shape(mask_img_input, rgb_img)
    if erode_boundary:
        kernel = np.ones((erod_boundary_kernel_size, erod_boundary_kernel_size),np.uint8)

    xyz_list = []
    new_depth_imgs = []
    for mask_idx in range(1, first_frame_annotation.max() + 1):
        masked_depth_img = depth_img.copy()
        binary_mask = np.where(mask_img == mask_idx, 1, 0)

        if erode_boundary:
            binary_mask = binary_erosion(binary_mask, kernel)

        masked_depth_img[binary_mask == 0] = -1

        if np.sum(binary_mask) == 0:
            if prev_xyz is None:
                points = np.zeros((cfg.datasets.max_points, 3))
            else:
                points = prev_xyz[mask_idx - 1]
        else:
            o3d_pc = O3DPointCloud(max_points=cfg.datasets.max_points)
            o3d_pc.create_from_depth(masked_depth_img, intrinsic_matrix)
            o3d_pc.transform(original_extrinsic_matrix)
            if (o3d_pc.get_num_points() == 0):
                raise ValueError("No points in the point cloud")
            
            new_masked_depth_img, z_max = get_transformed_depth_img(
                point_cloud=o3d_pc.get_points(),
                camera_intrinsics=intrinsic_matrix,
                new_camera_extrinsics=new_extrinsic_matrix,
                camera_width=rgb_img.shape[0],
                camera_height=rgb_img.shape[1],
            )
            # o3d_pc.preprocess(use_rgb=False)
            new_o3d_pc = O3DPointCloud(max_points=cfg.datasets.max_points)
            new_o3d_pc.create_from_depth(new_masked_depth_img,
                                         intrinsic_matrix,
                                         depth_trunc=z_max,
                                        #  depth_trunc=z.max() * alpha * 0.001
                                         )
            new_o3d_pc.transform(new_extrinsic_matrix)
            new_o3d_pc.preprocess(use_rgb=False)
            points = new_o3d_pc.get_points()
        
        xyz_list.append(points)
        # new_depth_imgs.append(new_masked_depth_img)

    return np.stack(xyz_list, axis=0) # , new_depth_imgs

def object_pcd_generation(
    cfg,
    dataset_name,
    mask_dataset_name,
    pcd_dataset_name,
    annotation_folder,
    camera_name="agentview",
    is_real_robot=True,
    erode_boundary=False,
    ):
    frist_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)

    # this is to map augmented demo to original demo for some values that are not changed, such as ee_states, joint_states, etc.
    aug_demo_mapping = {}
    with h5py.File(dataset_name, 'r') as dataset, \
         h5py.File(mask_dataset_name, 'r') as mask_dataset, \
         h5py.File(pcd_dataset_name, 'w') as pcd_dataset:
        
        pcd_dataset.create_group("data")
        intrinsic_matrix_dict = json.loads(dataset["data"].attrs["camera_intrinsics"])
        intrinsic_matrix = np.array(intrinsic_matrix_dict[camera_name])

        max_num_demo = len(dataset["data"].keys())

        # video_writer = VideoWriter(annotation_folder, "new_depth.mp4", fps=40.0, save_video=True)

        for demo in tqdm(dataset["data"].keys()):
            imgs = dataset[f"data/{demo}/obs/{camera_name}_rgb"][()]
            depth_imgs = dataset[f"data/{demo}/obs/{camera_name}_depth"][()]
            masks = mask_dataset[f"data/{demo}/obs/{camera_name}_masks"][()]
            extrinsics = dataset[f"data/{demo}/obs/{camera_name}_extrinsics"][()]
            aug_extrinsics = {}
            aug_episode_xyz = {}
            aug_episode_depth = {}
            for aug_rotation in cfg.datasets.aug.rotations:
                for angle in [aug_rotation, -aug_rotation]:
                    aug_episode_xyz[angle] = []
                    aug_episode_depth[angle] = []

            episode_xyz = []
            for img_idx in range(len(imgs)):
                xyz = object_pcd_fn(
                    cfg=cfg,
                    rgb_img_input=imgs[img_idx],
                    depth_img_input=depth_imgs[img_idx],
                    mask_img_input=masks[img_idx],
                    intrinsic_matrix=intrinsic_matrix,
                    extrinsic_matrix=extrinsics[img_idx],
                    first_frame_annotation=first_frame_annotation,
                    erode_boundary=erode_boundary,
                    is_real_robot=is_real_robot,
                    prev_xyz=episode_xyz[-1] if len(episode_xyz) > 0 else None,
                )
                episode_xyz.append(xyz)

                for angle in aug_episode_xyz.keys():
                    aug_extrinsics[angle] = rotate_camera_pose(extrinsics[img_idx],
                                                                angle=angle,
                                                                point=cfg.datasets.aug.workspace_center)

                for angle in aug_extrinsics.keys():
                    aug_xyz = object_pcd_aug_fn(
                    cfg=cfg,
                    rgb_img_input=imgs[img_idx],
                    depth_img_input=depth_imgs[img_idx],
                    mask_img_input=masks[img_idx],
                    intrinsic_matrix=intrinsic_matrix,
                    original_extrinsic_matrix=extrinsics[img_idx],
                    new_extrinsic_matrix=aug_extrinsics[angle],
                    first_frame_annotation=first_frame_annotation,
                    erode_boundary=erode_boundary,
                    is_real_robot=is_real_robot,
                    prev_xyz=aug_episode_xyz[angle][-1] if len(aug_episode_xyz[angle]) > 0 else None,
                    )
                    aug_episode_xyz[angle].append(aug_xyz)
                    # aug_episode_depth[angle].append(np.concatenate([depth_to_rgb(img) for img in new_depth_imgs], axis=0))

                
            # for angle in aug_episode_depth.keys():
            #     for new_depth_img in aug_episode_depth[angle]:
            #         video_writer.append_image(new_depth_img)
            episode_xyz = np.stack(episode_xyz, axis=0)
            for angle in aug_episode_xyz.keys():
                aug_episode_xyz[angle] = np.stack(aug_episode_xyz[angle], axis=0)
            #     print(aug_episode_xyz[angle].shape)
            # print(episode_xyz.shape)

            demo_idx = int(demo.split("_")[-1])
            pcd_dataset.create_dataset(f"data/{demo}/obs/xyz", data=episode_xyz)
            pcd_dataset.create_dataset(f"data/{demo}/obs/depth", data=imgs)
            for i, angle in enumerate(aug_episode_xyz.keys()):
                new_demo = f"demo_{max_num_demo * (i+1) + demo_idx}"
                pcd_dataset.create_dataset(f"data/{new_demo}/obs/xyz", data=aug_episode_xyz[angle])
                aug_demo_mapping[new_demo] = demo
        print(len(pcd_dataset["data"].keys()))
        pcd_dataset.attrs["aug_demo_mapping"] = json.dumps(aug_demo_mapping)
    return aug_demo_mapping

def pcd_grouping(
        cfg,
        pcd_dataset_name,
        pcd_grouped_dataset_name,
        is_real_robot=True
        ):
    """Given a point cloud, create groups of point clouds"""
    # parallel_process(pcd_dataset_name, cfg, pcd_grouped_dataset_name)
    num_group = cfg.datasets.num_group
    group_size = cfg.datasets.group_size
    group_divider = Group(num_group=num_group, group_size=group_size).cuda()

    with h5py.File(pcd_dataset_name, 'r') as pcd_dataset, \
        h5py.File(pcd_grouped_dataset_name, 'w') as pcd_grouped_dataset:
        
        pcd_grouped_dataset.create_group("data")

        for demo_key in tqdm(pcd_dataset["data"].keys()):
            xyz_sequence = pcd_dataset[f"data/{demo_key}/obs/xyz"][()]
            normalized_xyz_sequence = normalize_pcd(xyz_sequence, max_array=cfg.datasets.max_array, min_array=cfg.datasets.min_array)
            B, N, D = xyz_sequence.shape[:-1]

            xyz_tensor = torch.from_numpy(normalized_xyz_sequence).cuda().float()
            xyz_tensor = rearrange(xyz_tensor, "b n d t-> (b n) d t")
            neighborhood, centers = group_divider(xyz_tensor)
            centers = centers.unsqueeze(-2)
            neighborhood = rearrange(neighborhood, "(b n) g d t -> b (n g) d t", b=B, n=N)
            centers = rearrange(centers, "(b n) g d t -> b (n g) d t", b=B, n=N)

            pcd_grouped_dataset.create_group(f"data/{demo_key}/obs")
            pcd_grouped_dataset[f"data/{demo_key}/obs"].create_dataset(f"neighborhood_{num_group}_{group_size}", data=neighborhood.cpu().numpy())
            pcd_grouped_dataset[f"data/{demo_key}/obs"].create_dataset(f"centers_{num_group}_{group_size}", data=centers.cpu().numpy())

            pcd_grouped_dataset[f"data/{demo_key}/obs"].create_dataset(f"xyz", data=xyz_sequence)

# def worker_fn(xyz_sequence_dict, cfg, demo_keys_chunk, tmp_file):
#     with h5py.File(tmp_file, 'w') as tmp_hdf5:
        
#         tmp_hdf5.create_group("data")
#         num_group = cfg.datasets.num_group
#         group_size = cfg.datasets.group_size
#         group_divider = Group(num_group=num_group, group_size=group_size).cuda()

#         for demo_key in demo_keys_chunk:
#             # xyz_sequence = pcd_dataset[f"data/{demo_key}/obs/xyz"][()]
#             xyz_sequence = xyz_sequence_dict[demo_key]
#             normalized_xyz_sequence = normalize_pcd(xyz_sequence, max_array=cfg.datasets.max_array, min_array=cfg.datasets.min_array)
#             B, N, D = xyz_sequence.shape[:-1]

#             xyz_tensor = torch.from_numpy(normalized_xyz_sequence).cuda().float()
#             xyz_tensor = rearrange(xyz_tensor, "b n d t-> (b n) d t")
#             neighborhood, centers = group_divider(xyz_tensor)
#             centers = centers.unsqueeze(-2)
#             neighborhood = rearrange(neighborhood, "(b n) g d t -> b (n g) d t", b=B, n=N)
#             centers = rearrange(centers, "(b n) g d t -> b (n g) d t", b=B, n=N)

#             tmp_hdf5.create_group(f"data/{demo_key}/obs")
#             tmp_hdf5[f"data/{demo_key}/obs"].create_dataset(f"neighborhood_{num_group}_{group_size}", data=neighborhood.cpu().numpy())
#             tmp_hdf5[f"data/{demo_key}/obs"].create_dataset(f"centers_{num_group}_{group_size}", data=centers.cpu().numpy())

# def merge_temp_files(temp_files, output_file):
#     with h5py.File(output_file, 'w') as final_hdf5:
#         final_hdf5.create_group("data")

#         for tmp_file in temp_files:
#             with h5py.File(tmp_file, 'r') as tmp_hdf5:
#                 for demo_key in tmp_hdf5["data"].keys():
#                     final_hdf5.copy(tmp_hdf5[f"data/{demo_key}"], f"data/{demo_key}")
#             os.remove(tmp_file)

# def parallel_process(pcd_dataset_name, cfg, pcd_grouped_dataset_name):
#     if get_start_method(allow_none=True) != 'spawn':
#         set_start_method('spawn')
#     xyz_sequence_dict = {}
#     with h5py.File(pcd_dataset_name, 'r') as pcd_dataset:
#         demo_keys = list(pcd_dataset["data"].keys())

#         for demo_key in demo_keys:
#             xyz_sequence_dict[demo_key] = pcd_dataset[f"data/{demo_key}/obs/xyz"][()]
        
#     # Split demo_keys into smaller chunks
#     num_processes = 10
#     chunk_size = len(demo_keys) // num_processes
#     chunks = [demo_keys[i:i + chunk_size] for i in range(0, len(demo_keys), chunk_size)]

#     # Generate temporary file names for each process
#     temp_files = [f"temp_{i}.hdf5" for i in range(num_processes)]

#     # Start worker processes
#     with Pool(num_processes) as p:
#         p.starmap(worker_fn, [(xyz_sequence_dict, cfg, chunk, tmp_file) for chunk, tmp_file in zip(chunks, temp_files)])

#     # Merge temporary files into the final hdf5 file
#     merge_temp_files(temp_files, pcd_grouped_dataset_name)