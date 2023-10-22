from ctypes import resize
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
from functools import partial

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
from groot_imitation.groot_algo import toggle_data_modality_processing, GROOT_ROOT_PATH
from groot_imitation.groot_algo.xmem_tracker import XMemTracker
from groot_imitation.groot_algo.misc_utils import add_palette_on_mask, get_annotation_path, get_first_frame_annotation, get_first_frame_annotation_from_dataset, overlay_xmem_mask_on_image, mask_to_rgb, resize_image_to_same_shape, normalize_pcd, get_palette
from groot_imitation.groot_algo.o3d_modules import O3DPointCloud, convert_convention
from groot_imitation.groot_algo.eval_utils import raw_real_obs_to_tensor_obs
from groot_imitation.groot_algo.dataset_preprocessing.pcd_generation import object_pcd_fn

from groot_imitation.groot_algo.groot_transformer import GROOTSingleTask
from robomimic.utils.obs_utils import Modality
from groot_imitation.groot_algo import PcdModality, NormalizedPcdModality,WristDepthModality, GroupedPcdModality, normalize_real_robot_point_cloud, GROOT_Real_Robot_Benchmark

from groot_imitation.groot_algo.sam_operator import SAMOperator
from groot_imitation.groot_algo.dino_features import DinoV2ImageProcessor
from segmentation_correspondence_model.scm import SegmentationCorrespondenceModel

from real_robot_scripts.groot_img_utils import ImageProcessor
from real_robot_scripts.real_robot_utils import RealRobotObsProcessor, RolloutLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Get all the masks for the demonstration dataset')
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    # test
    # parser.add_argument('--test', action='store_true', help='eval')
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

    experiment_options = ["canonical", 
                          "distractions", 
                          "camera", 
                          "new_instances", 
                          "camera_new_instances", 
                          "camera_distractions"]

    exp_cfg = YamlConfig(args.experiment_config).as_easydict()

    args.experiment_name = exp_cfg.experiment_name
    if exp_cfg.experiment_name != "new_instances":
        args.new_instance_idx = -1
    
    assert(args.experiment_name in experiment_options), f"Please specify from {experiment_options}"

    obs_cfg = YamlConfig("real_robot_scripts/real_robot_observation_cfg_example.yml").as_easydict()

    #  _                    _  
    # | |    ___   __ _  __| | 
    # | |   / _ \ / _` |/ _` | 
    # | |__| (_) | (_| | (_| | 
    # |_____\___/ \__,_|\__,_| 
                            
    #   ____ _               _                _       _   
    #  / ___| |__   ___  ___| | ___ __   ___ (_)_ __ | |_ 
    # | |   | '_ \ / _ \/ __| |/ / '_ \ / _ \| | '_ \| __|
    # | |___| | | |  __/ (__|   <| |_) | (_) | | | | | |_ 
    #  \____|_| |_|\___|\___|_|\_\ .__/ \___/|_|_| |_|\__|
    #                            |_|                      

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
    task_id = cfg.task_id
    benchmark = get_benchmark(cfg.benchmark_name)()
    dataset_name = os.path.join(cfg.folder)

    try:
        _ , shape_meta = get_dataset(
                dataset_path=dataset_name,
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len)
    except:
        print(f"[error] failed to load task {task_id} name {benchmark.get_task_names()[task_id]}") 

    #  _                    _   ____                       
    # | |    ___   __ _  __| | |  _ \  ___ _ __ ___   ___  
    # | |   / _ \ / _` |/ _` | | | | |/ _ \ '_ ` _ \ / _ \ 
    # | |__| (_) | (_| | (_| | | |_| |  __/ | | | | | (_) |
    # |_____\___/ \__,_|\__,_| |____/ \___|_| |_| |_|\___/ 
                                                        
    #     _                      _        _   _             
    #    / \   _ __  _ __   ___ | |_ __ _| |_(_) ___  _ __  
    #   / _ \ | '_ \| '_ \ / _ \| __/ _` | __| |/ _ \| '_ \ 
    #  / ___ \| | | | | | | (_) | || (_| | |_| | (_) | | | |
    # /_/   \_\_| |_|_| |_|\___/ \__\__,_|\__|_|\___/|_| |_|

    first_frame, first_frame_annotation = get_first_frame_annotation_from_dataset(dataset_name)

    #  ____  _             _      ____
    # / ___|| |_ __ _ _ __| |_   / ___|__ _ _ __ ___   ___ _ __ __ _
    # \___ \| __/ _` | '__| __| | |   / _` | '_ ` _ \ / _ \ '__/ _` |
    #  ___) | || (_| | |  | |_  | |__| (_| | | | | | |  __/ | | (_| |
    # |____/ \__\__,_|_|   \__|  \____\__,_|_| |_| |_|\___|_|  \__,_|

    # We abstract away many details in processing camera images in the class RealRobotObsProcessor. More details please refer to the class.
    obs_processor = RealRobotObsProcessor(obs_cfg,
                                          processor_name="ImageProcessor")
    prev_points = []
    obs_processor.load_intrinsic_matrix()
    obs_processor.load_extrinsic_matrix()

    # Segment Correspondence Model
    #  ____   ____ __  __
    # / ___| / ___|  \/  |
    # \___ \| |   | |\/| |
    #  ___) | |___| |  | |
    # |____/ \____|_|  |_|


    if args.new_instance_idx < 0:
        is_new_instance = False
    else:
        is_new_instance = True

    if is_new_instance:
        new_instance_annotation_folder = get_annotation_path(dataset_name)
        new_instance_annotation_folder = os.path.join(new_instance_annotation_folder, f"evaluation_{args.new_instance_idx}")
        os.makedirs(new_instance_annotation_folder, exist_ok=True)
        if os.path.exists(os.path.join(new_instance_annotation_folder, "frame.jpg")) and os.path.exists(os.path.join(new_instance_annotation_folder, "frame_annotation.png")):
            first_frame, first_frame_annotation = get_first_frame_annotation(new_instance_annotation_folder)
            cv2.imshow("", first_frame)
            cv2.waitKey(0)
            print(colored("Skipping since this instance is already annotated", "yellow"))
        else:
            dinov2 = DinoV2ImageProcessor()
            sam_operator = SAMOperator()
            sam_operator.init()
            scm_module = SegmentationCorrespondenceModel(dinov2=dinov2, sam_operator=sam_operator)

            xmem_input_size = first_frame.shape[:2]
            new_first_frame = convert_convention(obs_processor.get_real_robot_img_obs()["agentview_rgb"])
            first_frame = resize_image_to_same_shape(first_frame, new_first_frame)
            first_frame_annotation = resize_image_to_same_shape(first_frame_annotation, first_frame)

            print(first_frame.shape, first_frame_annotation.shape, new_first_frame.shape)
            new_first_frame_annotation = scm_module(new_first_frame, first_frame, first_frame_annotation)
            new_first_frame_annotation = resize_image_to_same_shape(new_first_frame_annotation, new_first_frame)

            new_first_frame_annotation[first_frame_annotation == first_frame_annotation.max()] = first_frame_annotation.max()

            new_first_frame = resize_image_to_same_shape(new_first_frame, reference_size=xmem_input_size)
            new_first_frame_annotation = resize_image_to_same_shape(np.array(new_first_frame_annotation), reference_size=xmem_input_size)

            new_first_frame_annotation = Image.fromarray(new_first_frame_annotation)
            new_first_frame_annotation.putpalette(get_palette(palette="davis"))
            # Write the frame of new instancs into the file.
            cv2.imwrite(os.path.join(new_instance_annotation_folder, "frame.jpg"), new_first_frame)
            new_first_frame_annotation.save(os.path.join(new_instance_annotation_folder, "frame_annotation.png"))

            first_frame = new_first_frame
            first_frame_annotation = np.array(new_first_frame_annotation)

    # This is a placeholder as LIBERO codebase requires it.
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

    # __  __  ____                
    # \ \/ /  \/  | ___ _ __ ___  
    #  \  /| |\/| |/ _ \ '_ ` _ \ 
    #  /  \| |  | |  __/ | | | | |
    # /_/\_\_|  |_|\___|_| |_| |_|


    xmem_tracker = XMemTracker(xmem_checkpoint=os.path.join(GROOT_ROOT_PATH, 'third_party/xmem_checkpoints/XMem.pth'), 
                               device='cuda:0')
    xmem_tracker.clear_memory()


    # Set observation processor for point clouds
    def normalize_real_robot_point_cloud_obs_processor(obs, max_array, min_array):
        start_dims = np.arange(len(obs.shape) - 3).tolist()
        # print(f"normalizing with {max_array}")
        obs = normalize_pcd(obs, max_array, min_array)
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])
            
    normalize_pcd_func = partial(normalize_real_robot_point_cloud_obs_processor, max_array=cfg.datasets.max_array, min_array=cfg.datasets.min_array)

    toggle_data_modality_processing(normalize_pcd_func)
    # task_emb = benchmark.get_task_embs(cfg.task_id)
    

    #  ___       _ _     ____                           
    # |_ _|_ __ (_) |_  |  _ \  ___  _____  ___   _ ___ 
    #  | || '_ \| | __| | | | |/ _ \/ _ \ \/ / | | / __|
    #  | || | | | | |_  | |_| |  __/ (_) >  <| |_| \__ \
    # |___|_| |_|_|\__| |____/ \___|\___/_/\_\\__, |___/
    #                                         |___/     

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

    xmem_tracker.track(first_frame, first_frame_annotation)
    print(first_frame.shape, first_frame_annotation.shape)
    count = 0
    write_pcd = O3DPointCloud()

    #  ____       _ _             _   
    # |  _ \ ___ | | | ___  _   _| |_ 
    # | |_) / _ \| | |/ _ \| | | | __|
    # |  _ < (_) | | | (_) | |_| | |_ 
    # |_| \_\___/|_|_|\___/ \__,_|\__|
    rollout_logger = RolloutLogger()
    
    with torch.no_grad():
        while count < args.max_steps:

            # 1. Spacemouse actions

            spacemouse_input, _ = input2action(
                device=device,
                controller_type=controller_type,
            )
            if spacemouse_input is None:
                break

            # Skip if robot states have never been received so far
            if len(robot_interface._state_buffer) == 0:
                continue

            # 2. Get the latest state of the robot (inlucding the gripper)
            last_state = robot_interface._state_buffer[-1]
            last_gripper_state = robot_interface._gripper_state_buffer[-1]

            count += 1

            # 3. Get observations in the obs_processor
            obs_processor.get_real_robot_state(last_state, last_gripper_state)
            obs_processor.get_real_robot_img_obs()

            extrinsic_matrix = obs_processor.get_extrinsic_matrix("agentview")
            intrinsic_matrix = obs_processor.get_intrinsic_matrix("agentview")

            obs = obs_processor.obs

            # 4. Track the objects
            curr_img = resize_image_to_same_shape(obs["agentview_rgb"], first_frame)
            mask = xmem_tracker.track(curr_img)

            # Make sure the xmem prediction is stable at the first frame before the robot starts moving.
            if count < 20:
                continue
            if is_new_instance:
                if count < 10:
                    continue

            # Visualize tracking results
            overlay_image = overlay_xmem_mask_on_image(curr_img, mask)
            cv2.imshow("overlay_image", overlay_image)
            cv2.waitKey(10)

            colored_mask = mask_to_rgb(mask)
            cv2.imshow("mask", colored_mask)
            cv2.waitKey(10)

            mask = resize_image_to_same_shape(mask, reference_size=obs["agentview_rgb"].shape[:2])

            # 5. Get the point cloud from RGB-D observations
            xyz_array = object_pcd_fn(
                obs_cfg,
                rgb_img_input=obs["agentview_rgb"],
                depth_img_input=obs["agentview_depth"],
                mask_img_input=mask,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                first_frame_annotation=first_frame_annotation,
                erode_boundary=obs_cfg.datasets.erode,
                is_real_robot=True,
                prev_xyz=prev_points[-1] if len(prev_points) > 0 else None,
            )

            prev_points.append(xyz_array)

            #  Default observation modality is NormalizedPcdModality, which will take in the point clouds and normalize the scale of point cloud positions
            obs["xyz"] = xyz_array

            # 6. Get the observation tensor
            data = raw_real_obs_to_tensor_obs(obs, task_emb, cfg)

            # print(obs["xyz"][1].mean(axis=0))
            # print(obs["xyz"][1].max(axis=0))
            # print(obs["xyz"][1].min(axis=0))


            # print("-----------------")
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].mean(axis=0))
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].max(axis=0))
            # print(data["obs"]["xyz"].permute(0, 1, 3, 2)[0][0].min(axis=0))
            # for i, color in [(0, [1, 0, 0]), (1, [0, 1, 0])]:
            #     new_write_pcd = O3DPointCloud()
            #     new_write_pcd.create_from_points(obs["xyz"][i:i+1].reshape(-1, 3))
            #     new_write_pcd.pcd.paint_uniform_color(color)
            #     write_pcd.merge(new_write_pcd)
            #     if first_frame_annotation.max() == 2:
            #         break
            # break

            # 7. Infer the actions
            action = algo.policy.get_action(data)

            # 8. Send the actions to the robot
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            # log rollout info
            rollout_logger.log_info(obs, action)

        # write_pcd.save(f"real_robot_scripts/test_point_cloud.ply")

    #  ____                    ____        _        
    # / ___|  __ ___   _____  |  _ \  __ _| |_ __ _ 
    # \___ \ / _` \ \ / / _ \ | | | |/ _` | __/ _` |
    #  ___) | (_| |\ V /  __/ | |_| | (_| | || (_| |
    # |____/ \__,_| \_/ \___| |____/ \__,_|\__\__,_|

    print("Saving data")

    # Save the rollout info
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

    state_log_folder = str(folder/ run_experiment_str / f"state_log")
    os.makedirs(state_log_folder, exist_ok=True)
    rollout_logger.save(folder_name=state_log_folder)

    # Determine if saving the experiment run this time
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
    
    # Determine if the current rollout is successful
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
