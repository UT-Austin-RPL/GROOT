# Load model from a checkpoint (read all its configs as well)

import argparse
import sys
import os
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
import cv2

from pathlib import Path
import pprint
from tqdm import trange
from PIL import Image
from easydict import EasyDict

from termcolor import colored

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import get_algo_class
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset
# from libero.lifelong.metric import evaluate_loss, evaluate_success, raw_obs_to_tensor_obs
from libero.lifelong.utils import control_seed, safe_device, torch_load_model, \
                           NpEncoder, compute_flops

from libero.lifelong.main import get_task_embs
# from libero.lifelong.metric import raw_obs_to_tensor_obs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import robosuite.utils.transform_utils as T

import time

from kaede_utils.visualization_utils.video_utils import KaedeVideoWriter

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.structures import Boxes, pairwise_iou
from detectron2.config import get_cfg

import init_path
from vos_3d_algo import PcdModality, VOS_3D_Benchmark
from vos_3d_algo.env_wrapper import ViewChangingOffScreenRenderEnv
from vos_3d_algo.baselines import *
from vos_3d_algo.vos_3d_transformer import VOS3DSingleTask
from robomimic.utils.obs_utils import Modality
from vos_3d_algo.xmem_tracker import XMemTracker
from vos_3d_algo.misc_utils import get_annotation_path, get_first_frame_annotation
from vos_3d_algo.eval_utils import raw_obs_to_tensor_obs
from vos_3d_algo.o3d_modules import convert_convention, O3DPointCloud

def get_experiment_info(cfg):
    print(colored("Using point cloud augmentation? ", "yellow"), cfg.policy.pcd_aug.network_kwargs.enabled)
    if "pcd" in cfg.data.obs.modality:
        print(colored("Using point cloud observations : ", "yellow"), cfg.data.obs.modality.pcd)
    else:
        print("Using normalized point clouds")
    print(colored("Embed size : ", "yellow"), cfg.policy.embed_size)

def load_detection_model(model_type="fcnn", verbose=False):
    cfg = get_cfg()

    if model_type == "fcnn":
        config_file_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    elif model_type == "mask_rcnn":
        config_file_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file_name))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_name)    
    cfg.MODEL.META_ARCHITECTURE = 'ProposalNetwork'
    cfg.MODEL.POST_NMS_TOPK_TEST = 128
    cfg.MODEL.RPN.NMS_THRESH = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    predictor = DefaultPredictor(cfg)
    if verbose:
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(cfg)
    return predictor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--experiment_dir', type=str, default="experiments")
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--multiview', action='store_true')
    parser.add_argument('--camera-idx', type=int, default=-1)
    parser.add_argument('--training-mode', action='store_true')
    parser.add_argument('--checkpoint-idx', type=int, default=-1)
    parser.add_argument('--num-eval', type=int, default=-1)
    parser.add_argument('--different-scene', action='store_true')
    args = parser.parse_args()
    args.device_id = "cuda:"+str(args.device_id)

    return args




def main():
    args = parse_args()
    run_folder = args.checkpoint_dir
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
    predictor = load_detection_model("fcnn")

    # cfg.policy.pcd_encoder.network_kwargs.masked_encoder_cfg.mask_ratio = 0
    # cfg.policy.pcd_encoder.network_kwargs.masked_encoder_cfg.mask_ratio = 0.
    task_id = cfg.task_id # margs.task_id
    benchmark = get_benchmark(cfg.benchmark_name)()
    dataset_name = os.path.join(cfg.folder,
                                benchmark.get_task_demonstration(task_id))
    try:
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=dataset_name,
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len)
    except:
        print(f"[error] failed to load task {task_id} name {benchmark.get_task_names()[task_id]}") 

    num_point_per_cloud = cfg.shape_meta["all_shapes"]["viola_xyz"][-1]
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(10, cfg), cfg.device)

    print(get_experiment_info(cfg))
    # algo.policy.previous_mask = previous_mask

    algo.policy.load_state_dict(sd)

    algo.eval()

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    # task_embs = get_task_embs(cfg, descriptions)
    task_embs = torch.ones((len(descriptions), 1))

    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(cfg.task_id)

    ### ======================= start evaluation ============================

    if not args.training_mode:
        eval_str = "_eval"
    else:
        eval_str = ""
    if args.multiview:
        save_folder = os.path.join(run_folder, f"stats_multi_view{eval_str}")
    else:
        save_folder = os.path.join(run_folder, f"stats{eval_str}")
    os.makedirs(save_folder, exist_ok=True)   

    eval_stats = {
        "success": [],
    }
    pp = pprint.PrettyPrinter(indent=4)

    camera_idx = args.camera_idx

    # else:
    video_folder = os.path.join(run_folder, f"videos_task_{task_id}")
    if args.checkpoint_idx > -1:
        video_folder += f"_checkpoint_{args.checkpoint_idx}"
    if args.different_scene:
        video_folder += "_different_scene"
    if camera_idx > -1:
        video_folder += f"{eval_str}_camera_{camera_idx}"
    os.makedirs(video_folder, exist_ok=True)

    # Set up the tracking
    annotations_folder = get_annotation_path(dataset_name)
    first_frame, first_frame_annotation = get_first_frame_annotation(annotations_folder)
    first_frame = np.ascontiguousarray(first_frame)
    
    overlay_video_writer = KaedeVideoWriter(video_folder, video_name="overlay_video.mp4", save_video=args.save_videos)

    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
            "camera_idx": camera_idx,
            "training_mode": args.training_mode,
            "camera_depths": True,
        }
        if args.different_scene:
            if "KITCHEN" in task.bddl_file:
                new_scene_xml = "scenes/libero_kitchen_tabletop_dark_table_style.xml"
                if args.camera_idx == -2:
                    new_scene_xml = "scenes/libero_kitchen_tabletop_visual_distractions.xml"
                elif args.camera_idx == -3:
                    new_scene_xml = "scenes/libero_kitchen_tabletop_light_off.xml"
            env_args.update({"scene_xml": new_scene_xml})
        env_creation = False
        count = 0
        env = ViewChangingOffScreenRenderEnv(**env_args)
        print("env created")

        init_states_path = os.path.join(cfg.init_states_folder,
                                        task.problem_folder,
                                        task.init_states_file)
        init_states = torch.load(init_states_path)
        print("init state loaded")
        # indices = np.arange(env_num) % init_states.shape[0]
        # init_states_ = init_states[indices]
        num_success = 0
        cfg.eval.n_eval = 20 if args.num_eval == -1 else args.num_eval
        for i in trange(cfg.eval.n_eval):
            env.reset()
            env.seed(cfg.seed)
            algo.reset()

            # reset the video writer for correct visualization
            video_writer.reset()

            done = False
            steps = 0
            init_states_ = init_states[i]

            obs = env.set_init_state(init_states_)
            task_emb = benchmark.get_task_emb(args.task_id)

            for _ in range(5): # simulate the physics without any actions
                obs, _, _, _ = env.step([0.] * 7)

            prev_points = []
            prev_colors = []
            with torch.no_grad():
                for steps in trange(cfg.eval.max_steps):
                # while steps < cfg.eval.max_steps:
                    # steps += 1
                    with Timer() as t:

                        # Process Depth
                        obs["robot0_eye_in_hand_depth"] = (get_real_depth_map(env.sim, obs["robot0_eye_in_hand_depth"]) * 1000).astype(np.uint16).astype(np.float32)

                        obs["ee_states"] = np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))

                        bbox_outputs = predictor(obs["agentview_image"])
                        bboxes = bbox_outputs["proposals"].proposal_boxes.tensor[:20].cpu().numpy()


                        color = convert_convention(obs["agentview_image"])
                        depth = get_real_depth_map(env.sim, convert_convention(obs["agentview_depth"])) * 1000

                        xyz_list = []
                        rgb_list = []
                        intrinsic_matrix = get_camera_intrinsic_matrix(env.sim, "agentview", color.shape[0], color.shape[1])
                        extrinsic_matrix = get_camera_extrinsic_matrix(env.sim, "agentview")
                        
                        for bbox in bboxes:
                            mask = np.zeros((color.shape[0], color.shape[1])).astype(np.uint8)
                            masked_depth = depth.copy()
                            count += 1
                            bbox = bbox.astype(np.int32)
                            x0 = np.floor(bbox[0]).astype(np.int32)
                            y0 = np.floor(bbox[1]).astype(np.int32)
                            x1 = np.ceil(bbox[2]).astype(np.int32)
                            y1 = np.ceil(bbox[3]).astype(np.int32)
                            x0 = np.clip(x0, 0, color.shape[1] - 1)
                            y0 = np.clip(y0, 0, color.shape[0] - 1)
                            x1 = np.clip(x1, 0, color.shape[1] - 1)
                            y1 = np.clip(y1, 0, color.shape[0] - 1)
                                
                            mask[y0:y1, x0:x1] = 1

                            mask = convert_convention(mask)

                            binary_mask = np.where(mask==1, 1, 0)
                            masked_depth[binary_mask == 0] = -1

                            pcd = O3DPointCloud(max_points=512)
                            pcd.create_from_rgbd(color, masked_depth, intrinsic_matrix)
                            pcd.transform(extrinsic_matrix)

                            if pcd.get_num_points() == 0:
                                points = np.zeros((num_point_per_cloud, 3))
                            else:
                                pcd.preprocess()
                                points = pcd.get_points()
                            xyz_list.append(points)               

                    prev_points.append(xyz_list)
                    obs["viola_xyz"] = np.stack(xyz_list, axis=0)
                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    actions = algo.policy.get_action(data)
                    obs, reward, done, info = env.step(actions)
                    # video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")
                    video_writer.append_obs(obs, done, camera_name="agentview_image")

                    # check whether succeed
                    if done: break
                num_success += int(done)

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        time.sleep(3)
        eval_stats["success"].append(success_rate)

        if args.save_videos:
            print(colored(f"Videos are saved at {video_folder}", "yellow"))
    overlay_video_writer.save("overlay_video.mp4", flip=True)
    print(f"The whole process took: {t.get_elapsed_time()} sec")
    if args.checkpoint_idx == -1:
        torch.save(eval_stats, os.path.join(save_folder, f"stats_{args.camera_idx}_scene{args.different_scene}.pt"))
    else:
        torch.save(eval_stats, os.path.join(save_folder, f"stats_{args.camera_idx}_scene{args.different_scene}_{args.checkpoint_idx}.pt"))
    print(f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts")
    print(f"Results are saved at {save_folder}")

    print(eval_stats)

if __name__ == "__main__":
    main()
