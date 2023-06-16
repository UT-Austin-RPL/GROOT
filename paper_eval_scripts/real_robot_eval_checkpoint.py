import h5py
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import torch
from pathlib import Path

from termcolor import colored
from easydict import EasyDict

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import safe_device, torch_load_model
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset


from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.input_utils import input2action
from rpl_vision_utils.networking.camera_redis_interface import CameraRedisSubInterface

from kaede_utils.visualization_utils.video_utils import KaedeVideoWriter

import init_path
from vos_3d_algo import toggle_data_modality_processing
from vos_3d_algo.xmem_tracker import XMemTracker
from vos_3d_algo.misc_utils import get_annotation_path, get_first_frame_annotation, get_first_frame_annotation_from_dataset, overlay_xmem_mask_on_image, mask_to_rgb
from vos_3d_algo.o3d_modules import O3DPointCloud
from vos_3d_algo.eval_utils import raw_real_obs_to_tensor_obs

from vos_3d_algo.vos_3d_transformer import VOS3DSingleTask
from robomimic.utils.obs_utils import Modality
from vos_3d_algo import PcdModality, NormalizedPcdModality, VOS_3D_Benchmark, WristDepthModality, GroupedPcdModality, VOS_3D_Ablation_Augmentation_Benchmark, VOS_3D_Ablation_Grouping_Benchmark, normalize_real_robot_point_cloud


from real_robot_scripts.vos_3d_img_utils import ImageProcessor
from real_robot_scripts.real_robot_utils import RealRobotObsProcessor, RolloutLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Get all the masks for the demonstration dataset')
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    # test
    parser.add_argument('--test', action='store_true', help='eval')
    parser.add_argument('--is-demo', action='store_true', help='eval')
    parser.add_argument('--new-instance-idx', default=-1, type=int)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--checkpoint-idx', type=int, default=100)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--device-id', type=int, default=0)
    # max steps
    parser.add_argument('--max-steps', type=int, default=600)
    # experiment_name
    # parser.add_argument('--experiment-name', type=str, default="canonical")
    parser.add_argument('--experiment-config', type=str, required=True) # default="real_robot_scripts/eval_canonical.yaml")

    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)

    return args

def main():
    args = parse_args()
    run_folder = args.checkpoint_dir

    experiment_options = ["canonical", "distractions", "camera", "new_instances", "camera_new_instances", "camera_distractions"]

    exp_cfg = YamlConfig(args.experiment_config).as_easydict()

    args.experiment_name = exp_cfg.experiment_name
    args.test = exp_cfg.test
    args.new_instance_idx = int(exp_cfg.new_instance_idx)
    
    assert(args.experiment_name in experiment_options), f"Please specify from {experiment_options}"

    if args.test:
        assert(args.experiment_name in experiment_options[2:]), f"Testing needs"

    if args.test:
        obs_cfg = YamlConfig("real_robot_scripts/real_robot_test_cfg_example.yml").as_easydict()
    else:    
        obs_cfg = YamlConfig("real_robot_scripts/real_robot_observation_cfg_example.yml").as_easydict()
    try:
        if args.checkpoint_idx == -1:
            model_path = os.path.join(run_folder, f"task{args.task_id}_model.pth")
        else:
            model_path = os.path.join(run_folder, f"task{args.task_id}_model_{args.checkpoint_idx}.pth")
        sd, cfg, previous_mask = torch_load_model(model_path, map_location=args.device_id)
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")

    with open(os.path.join(run_folder, "config.json"), "r") as f:
        cfg = EasyDict(json.load(f))

    print(colored(f"loading checkpoint from {str(model_path)}", "green"))

    # cfg.policy.pcd_encoder.network_kwargs.masked_encoder_cfg.mask_ratio = 0
    task_id = cfg.task_id # margs.task_id
    benchmark = get_benchmark(cfg.benchmark_name)()
    if "libero" in cfg.folder:
        dataset_name = os.path.join(cfg.folder,
                                    benchmark.get_task_demonstration(task_id))
        dataset_name = "/".join(dataset_name.split("/")[dataset_name.split("/").index("datasets"):])
    else:
        dataset_name = os.path.join(cfg.folder)
    try:
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=dataset_name,
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len)
    except:
        print(f"[error] failed to load task {task_id} name {benchmark.get_task_names()[task_id]}") 

    if args.new_instance_idx < 0:
        is_new_instance = False
    else:
        is_new_instance = True

    if is_new_instance:
        new_instance_annotation_folder = get_annotation_path(dataset_name)
        new_instance_annotation_folder = os.path.join(new_instance_annotation_folder, f"evaluation_{args.new_instance_idx}")
        print(new_instance_annotation_folder)
        if os.path.exists(os.path.join(new_instance_annotation_folder, "frame.jpg")):
            first_frame, _ = get_first_frame_annotation(new_instance_annotation_folder)
            cv2.imshow("", first_frame)
            cv2.waitKey(0)
            print(colored("Skipping since this instance is already annotated", "yellow"))
        else:
            os.system(f"python real_robot_scripts/real_robot_sam_result.py --dataset {dataset_name} --new-instance-idx {args.new_instance_idx}")

        
    task_embs = torch.ones((1, 1))
    task_emb = task_embs[0]
    benchmark.set_task_embs(task_embs)

    print(shape_meta)
    num_point_per_cloud = cfg.shape_meta["all_shapes"]["xyz"][-1]

    # initialize algo
    cfg.device = args.device_id
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(10, cfg), cfg.device)
    algo.policy.load_state_dict(sd)
    algo.eval()

    # print(get_experiment_info(cfg))

    # TODO: Load run_camera_node automatically

    first_frame, first_frame_annotation = get_first_frame_annotation_from_dataset(dataset_name)

    if is_new_instance:
        annotation_folder = get_annotation_path(dataset_name)
        annotation_folder = os.path.join(annotation_folder, f"evaluation_{args.new_instance_idx}")
        new_first_frame, new_first_frame_annotation = get_first_frame_annotation(annotation_folder)

        new_first_frame = np.ascontiguousarray(new_first_frame)
        new_first_frame_annotation = np.ascontiguousarray(new_first_frame_annotation)
        new_first_frame_annotation[np.where(first_frame_annotation == first_frame_annotation.max())] = first_frame_annotation.max()
        # import pdb; pdb.set_trace()
        # mask_rgb = mask_to_rgb(new_first_frame_annotation)

        # overlay_image = overlay_xmem_mask_on_image(new_first_frame, mask_rgb)        
        # cv2.imshow("Checking the image", overlay_image)
        # cv2.waitKey(0)

        first_frame = new_first_frame
        first_frame_annotation = new_first_frame_annotation


    xmem_tracker = XMemTracker(xmem_checkpoint='xmem_checkpoints/XMem.pth', device='cuda:0')

    # NormalizedPcdModality.set_obs_processor(normalize_real_robot_point_cloud)
    toggle_data_modality_processing(real_robot=True)
    
    obs_processor = RealRobotObsProcessor(obs_cfg,
                                          processor_name="ImageProcessor")
    prev_points = []
    obs_processor.load_intrinsic_matrix()
    obs_processor.load_extrinsic_matrix()

    # task_emb = benchmark.get_task_embs(cfg.task_id)
    
    xmem_tracker.clear_memory()

    # Initialize the robot interface
    interface_cfg="charmander.yml"
    controller_cfg_name="osc-pose-controller.yml"
    controller_type="OSC_POSE"
    robot_interface = FrankaInterface(os.path.join(config_root, interface_cfg))
    controller_cfg = YamlConfig(
        os.path.join(config_root, controller_cfg_name)
    ).as_easydict()
    # Initialize I/O device
    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()

    # The following should be always here

    algo.reset()
    # TODO: Load SAM and get the first frame annotation if required. Delete the sam model afterwards as it is very large.
    xmem_tracker.track(first_frame, first_frame_annotation)
    print(first_frame.shape, first_frame_annotation.shape)
    count = 0
    write_pcd = O3DPointCloud()

    rollout_logger = RolloutLogger()
    
    with torch.no_grad():
        while count < args.max_steps:
            spacemouse_input, _ = input2action(
                device=device,
                controller_type=controller_type,
            )
            if spacemouse_input is None:
                break
            if len(robot_interface._state_buffer) == 0:
                continue

            last_state = robot_interface._state_buffer[-1]
            last_gripper_state = robot_interface._gripper_state_buffer[-1]

            count += 1
            obs_processor.get_real_robot_state(last_state, last_gripper_state)
            obs_processor.get_real_robot_img_obs()

            obs = obs_processor.obs


            image = cv2.resize(obs["agentview_rgb"], (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA)
            mask = xmem_tracker.track(image)

            if count < 50:
                continue

            if is_new_instance:
                if count < 10:
                    continue

            overlay_image = overlay_xmem_mask_on_image(image, mask)
            cv2.imshow("overlay_image", overlay_image)
            cv2.waitKey(10)

            colored_mask = mask_to_rgb(mask)
            cv2.imshow("mask", colored_mask)
            cv2.waitKey(10)

            xyz_list = []
            extrinsic_matrix = obs_processor.get_extrinsic_matrix("agentview")
            intrinsic_matrix = obs_processor.get_intrinsic_matrix("agentview")

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.resize(mask, obs["agentview_rgb"].shape[:2], interpolation=cv2.INTER_NEAREST)
            for mask_idx in range(1, first_frame_annotation.max() + 1):
                masked_depth = obs["agentview_depth"].copy()
                binary_mask = np.where(mask==mask_idx, 1, 0)
                # binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
                # print(binary_mask.dtype)

                from scipy.ndimage import binary_erosion
                for _ in range(2):
                    binary_mask = binary_erosion(binary_mask, structure=kernel)
                masked_depth[binary_mask == 0] = -1

                if np.sum(binary_mask) == 0:
                    points = prev_points[-1][mask_idx-1]
                else:
                    pcd = O3DPointCloud(max_points=num_point_per_cloud)
                    
                    pcd.create_from_rgbd(obs["agentview_rgb"], masked_depth, intrinsic_matrix)
                    pcd.transform(extrinsic_matrix)

                    if pcd.get_num_points() == 0:
                        points = prev_points[-1][mask_idx-1]
                    else:
                        pcd.preprocess(use_rgb=False)

                    points = pcd.get_points()
                xyz_list.append(points)

            prev_points.append(xyz_list)
            obs["xyz"] = np.stack(xyz_list, axis=0)

            data = raw_real_obs_to_tensor_obs(obs, task_emb, cfg)

            print(obs["xyz"][1].mean(axis=0))
            print(obs["xyz"][1].max(axis=0))
            print(obs["xyz"][1].min(axis=0))


            print("-----------------")
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].mean(axis=0))
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].max(axis=0))
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].min(axis=0))
            for i, color in [(0, [1, 0, 0]), (1, [0, 1, 0])]:
                new_write_pcd = O3DPointCloud()
                new_write_pcd.create_from_points(obs["xyz"][i:i+1].reshape(-1, 3))
                new_write_pcd.pcd.paint_uniform_color(color)
                write_pcd.merge(new_write_pcd)
                if first_frame_annotation.max() == 2:
                    break
            # break

            action = algo.policy.get_action(data)
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            # log rollout info
            rollout_logger.log_info(obs, action)

        write_pcd.save(f"real_robot_scripts/test_point_cloud.ply")


    print("Saving data")

    log_file_folder = os.path.join("evaluation", dataset_name.split("/")[-1].replace(".hdf5", ""), args.experiment_name)

    folder = Path(log_file_folder)
    folder.mkdir(parents=True, exist_ok=True)
    experiment_id = 0
    for path in folder.glob('run*'):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split('run')[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1

    run_experiment_str = f"run{experiment_id}"
    os.makedirs(str(folder / run_experiment_str), exist_ok=True)
    # log_std_file = str(folder / run_experiment_str / f"std.log")
    # log_err_file = str(folder / run_experiment_str  / f"err.log")

    state_log_folder = str(folder/ run_experiment_str / f"state_log")
    os.makedirs(state_log_folder, exist_ok=True)
    rollout_logger.save(folder_name=state_log_folder)
    if args.is_demo:        
        exit()

    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass
    if not save:
        import shutil
        shutil.rmtree(f"{folder}/{run_experiment_str}")
        del device
        exit()
    
    valid_input = False
    success = 0
    while not valid_input:
        try:
            success = input("success or not? (enter 0 or 1)")
            success = int(success)
            if success not in [0, 1]:
                continue
            valid_input = True
        except:
            pass
    if save:
        if success:
            result_file = str(folder/ run_experiment_str / f"True.log")
        else:
            result_file = str(folder/ run_experiment_str / f"False.log")

        with open(result_file, "w") as f:
            pass
    print(colored(str(folder / run_experiment_str) + f" , success: {success}", "green"))
    del device


if __name__ == "__main__":
    main()
