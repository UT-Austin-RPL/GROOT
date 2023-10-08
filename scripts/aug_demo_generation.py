"""Get all the masks for the demonstration dataset"""
import h5py
import cv2
import os
import argparse
import numpy as np
from PIL import Image
import json
import torch

from tqdm import tqdm

from kaede_utils.visualization_utils.video_utils import KaedeVideoWriter
from kaede_utils.visualization_utils.image_utils import depth_to_rgb


import init_path
from vos_3d_algo.xmem_tracker import XMemTracker
from vos_3d_algo.misc_utils import get_annotation_path, get_first_frame_annotation
from vos_3d_algo.o3d_modules import O3DPointCloud, convert_convention
from vos_3d_algo.env_wrapper import rotate_real_camera

from scipy.ndimage import binary_erosion

def parse_args():
    args = argparse.ArgumentParser(description='Get all the masks for the demonstration dataset')
    args.add_argument('--dataset', type=str, default='data/demonstration_dataset.hdf5', help='path to the demonstration dataset')
    # visualize video for debugging
    args.add_argument('--save-video', action='store_true', help='visualize the video')
    args.add_argument('--max-points', type=int, default=512)
    args.add_argument('--rotation', type=int, default=5)
    # number of demonstration trajectories
    args.add_argument('--num-demos', type=int, default=50)
    args.add_argument('--erode', action='store_true', help='erode the mask')

    # see if it's real data
    args.add_argument('--real', action='store_true', help='real data')

    args.add_argument('--verbose', action='store_true', help='verbose')

    return args.parse_args()

def main():
    args = parse_args()
    max_points = args.max_points
    annotations_folder = get_annotation_path(args.dataset)
    first_frame, first_frame_annotation = get_first_frame_annotation(annotations_folder)

    dataset_name = args.dataset
    mask_dataset_name = os.path.join(annotations_folder, "masks.hdf5")
    pcd_dataset_name = os.path.join(annotations_folder, "aug_pcd.hdf5")

    rotation_value = int(args.rotation)
    # rotate_camera_matrices = {}

    # print(loaded_matrices["rotate_z"].keys())
    # rotate_camera_matrices[-rotation_value] = loaded_matrices["rotate_z"][-rotation_value]
    # rotate_camera_matrices[rotation_value] = loaded_matrices["rotate_z"][rotation_value]

    print(f"Augmenting by {args.rotation} degrees")

    with h5py.File(args.dataset, 'r') as dataset, \
         h5py.File(mask_dataset_name, 'r') as mask_dataset, \
         h5py.File(pcd_dataset_name, 'w') as pcd_dataset:
        count = 0
        pcd_dataset.create_group("data")
        for demo in tqdm(dataset["data"].keys()):
            # count += 1
            # if count < 25:
            #     continue            
            if args.verbose:
                print("processing demo: ", demo)
            images = dataset[f"data/{demo}/obs/agentview_rgb"][()]

            episode_xyz = []

            aug_episode_xyz = {}
            new_extrinsic_matrices = {}
            for key in [-rotation_value, rotation_value]:
                aug_episode_xyz[key] = []
                new_extrinsic_matrices[key] = []
            for img_idx in range(len(images)):

                camera_name = "agentview"
                intrinsic_matrix_dict = json.loads(dataset["data"].attrs["camera_intrinsics"])

                intrinsic_matrix = np.array(intrinsic_matrix_dict[camera_name])
                extrinsic_matrix = dataset[f"data/{demo}/obs/{camera_name}_extrinsics"][()][img_idx]
                color = convert_convention(images[img_idx], real_robot=True)
                depth = convert_convention(dataset[f"data/{demo}/obs/{camera_name}_depth"][()][img_idx], real_robot=True)
                mask = mask_dataset[f"data/{demo}/obs/{camera_name}_masks"][()][img_idx]        
                mask = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

                rotate_camera_matrices = {}
                for key in [-rotation_value, rotation_value]:
                    rotate_camera_matrices[key] = rotate_real_camera(extrinsic_matrix, angle=key, point=[0., 0.0, 0.0])
                    rotate_camera_matrices[key] = rotate_real_camera(rotate_camera_matrices[key], angle=0, point=[0.10, 0.0, 0.0])
                xyz_list = []
                aug_xyz_dict = {}
                for key in [-rotation_value, rotation_value]:
                    aug_xyz_dict[key] = []

                previous_binary_mask = {}

                if args.erode:
                    kernel = np.ones((3, 3), np.uint8)

                for mask_idx in range(1, first_frame_annotation.max() + 1):
                    masked_depth = depth.copy()
                    binary_mask = np.where(mask==mask_idx, 1, 0)

                    if args.erode:
                        for _ in range(2):
                            binary_mask = binary_erosion(binary_mask, structure=kernel)

                    if np.sum(binary_mask) < 100:
                        # We will not do erosion if the mask is shrunk too small
                        binary_mask = np.where(mask==mask_idx, 1, 0)

                    masked_depth[binary_mask == 0] = -1
                    for key in [-rotation_value, rotation_value]:
                        new_extrinsic_matrices[key].append(rotate_camera_matrices[key])

                    if np.sum(binary_mask) == 0:
                        """When occluded, the point cloud observation will be null. However, notice that the vos model will still keep track since it takes into account history memory."""
                        points = episode_xyz[-1][mask_idx-1]
                        # colors = episode_rgb[-1][mask_idx-1]
                        for key in rotate_camera_matrices.keys():
                            new_points = aug_episode_xyz[key][-1][mask_idx-1]
                            aug_xyz_dict[key].append(new_points)
                    else:
                        pcd = O3DPointCloud(max_points=max_points)
                        pcd.create_from_rgbd(color, masked_depth, intrinsic_matrix)
                        pcd.transform(extrinsic_matrix)

                        for key in rotate_camera_matrices.keys():
                            if len(pcd.get_points()) == 0:
                                pcd.create_from_points(aug_episode_xyz[key][-1][mask_idx-1])
                            points = pcd.get_points()
                            

                            new_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1)
                            new_points = np.linalg.inv(rotate_camera_matrices[key]) @ new_points.transpose()

                            fx = intrinsic_matrix[0, 0]
                            fy = intrinsic_matrix[1, 1]
                            cx = intrinsic_matrix[0, 2]
                            cy = intrinsic_matrix[1, 2]

                            z = new_points[2, :] * 1000
                            u = fx * new_points[0, :] * 1000 / z + cx
                            v = fy * new_points[1, :] * 1000 / z + cy
                            px = u.astype(np.int)
                            py = v.astype(np.int)
                            # Create a mask for valid indices

                            # Create an initial depth image with maximum values
                            camera_width, camera_height = color.shape[:2]
                            # try:
                            #     print(camera_width, camera_height, z.max())
                            # except:
                            #     import pdb; pdb.set_trace()
                            new_depth_img = np.ones((camera_height, camera_width)) * z.max()
                            sorted_indices = np.argsort(z)[::-1]

                            px = px[sorted_indices]
                            py = py[sorted_indices]
                            z = z[sorted_indices]
                            valid_indices = (px >= 0) & (px < camera_width) & (py >= 0) & (py < camera_height)
                            # Update depth image with valid indices
                            new_depth_img[py[valid_indices], px[valid_indices]] = np.minimum(
                                new_depth_img[py[valid_indices], px[valid_indices]], z[valid_indices]
                            )

                            # Convert depth image to uint16
                            new_depth_img = new_depth_img.astype(np.uint16)

                            new_pcd = O3DPointCloud(max_points=max_points)
                            alpha = 0.95
                            # while new_pcd.get_num_points() < 10:
                            new_pcd.create_from_depth(new_depth_img, intrinsic_matrix, depth_trunc=z.max() * alpha * 0.001)
                            # alpha = alpha + 0.05
                            new_pcd.transform(rotate_camera_matrices[key])
                            # cv2.imshow("old_depth", depth_to_rgb(depth))
                            # cv2.imshow("depth", depth_to_rgb(new_depth_img))
                            # cv2.waitKey(0)                            
                            if new_pcd.get_num_points() > 10 or img_idx == 0:
                                
                                if (new_pcd.get_num_points() == 0):
                                    alpha = 0.95
                                    while new_pcd.get_num_points() < 10 and alpha < 1.05:
                                        new_pcd.create_from_depth(new_depth_img, intrinsic_matrix, depth_trunc=z.max() * alpha * 0.001)
                                        alpha = alpha + 0.05
                                    new_pcd.transform(rotate_camera_matrices[key])                                    
                                    # cv2.imshow("old_depth", depth_to_rgb(depth))
                                    # cv2.imshow("depth", depth_to_rgb(new_depth_img))
                                    # cv2.waitKey(0)
                                    # import pdb; pdb.set_trace()

                                assert(new_pcd.get_num_points() != 0), f"This is serious error in processing data, happens to rotation {key}, getting {new_pcd.get_num_points()} points"

                                new_pcd.preprocess(use_rgb=False)
                                new_points = new_pcd.get_points()
                            else:
                                # new_points = aug_episode_xyz[key][-1][mask_idx-1]
                                new_points = pcd.get_points()
                                new_pcd.create_from_points(new_points)
                                new_pcd.preprocess(use_rgb=False)
                                new_points = new_pcd.get_points()
                                # print("Using previous points")
                                # print(new_points)

                            # cv2.imshow("old_depth", depth_to_rgb(masked_depth))
                            # cv2.imshow("depth", depth_to_rgb(new_depth_img))
                            # cv2.waitKey(0)
                            aug_xyz_dict[key].append(new_points)
                        pcd.preprocess(use_rgb=False)
                        # print(pcd.get_num_points())
                        # pcd.save(f"segmented_pcd_{mask_idx}.ply")
                        points = pcd.get_points()
                    xyz_list.append(points)
                xyz = np.stack(xyz_list, axis=0)
                for key in rotate_camera_matrices.keys():
                    aug_xyz = np.stack(aug_xyz_dict[key], axis=0)
                    aug_episode_xyz[key].append(aug_xyz)
                episode_xyz.append(xyz)
            episode_xyz = np.stack(episode_xyz, axis=0)
            # episode_rgb = np.stack(episode_rgb, axis=0)
            pcd_dataset.create_dataset(f"data/{demo}/obs/xyz", data=episode_xyz)
            demo_idx = int(demo.split("_")[-1])
            dataset.copy(dataset[f"data/{demo}/obs/{camera_name}_extrinsics"], pcd_dataset[f"data/{demo}/obs"], f"{camera_name}_extrinsics")     
            print(episode_xyz.reshape(-1, 3).max(axis=0), episode_xyz.reshape(-1, 3).min(axis=0))
            for i, key in enumerate(rotate_camera_matrices.keys()):
                try:
                    new_aug_episode_xyz = np.stack(aug_episode_xyz[key], axis=0)
                except:
                    import pdb; pdb.set_trace()
                new_demo_idx = demo_idx + (i+1) * int(args.num_demos)
                new_demo = f"demo_{new_demo_idx}"
                pcd_dataset.create_dataset(f"data/{new_demo}/obs/{camera_name}_extrinsics", data=np.stack(new_extrinsic_matrices[key], axis=0))
                pcd_dataset.create_dataset(f"data/{new_demo}/obs/xyz", data=new_aug_episode_xyz)
                print("New demo: ", new_aug_episode_xyz.reshape(-1, 3).max(axis=0), new_aug_episode_xyz.reshape(-1, 3).min(axis=0))
                if args.verbose:
                    print("saving demo: ", new_demo, " with xyz shape: ", new_aug_episode_xyz.shape)
            if args.verbose:
                print("In this episode, saved points of shape {} ".format(episode_xyz.shape))
            # break

if __name__ == '__main__':
    main()