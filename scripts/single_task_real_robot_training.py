"""In this script, we will launch a single training experiment. Adapted from LIBERO codebase."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from termcolor import colored

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, Benchmark, task_maps, register_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset,
                               get_dataset)
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (NpEncoder, compute_flops, control_seed, safe_device,
                            torch_load_model, get_task_embs)
from libero.libero.utils.time_utils import Timer
from functools import partial

import init_path

from vos_3d_algo import toggle_data_modality_processing
from vos_3d_algo.vos_3d_transformer import VOS3DSingleTask
from robomimic.utils.obs_utils import Modality
from vos_3d_algo import PcdModality, NormalizedPcdModality, WristDepthModality, GroupedPcdModality, VOS_3D_Real_Robot_Benchmark
from vos_3d_algo.misc_utils import normalize_pcd

def create_experiment_dir(cfg):
    prefix = "experiments_real"

    task_name = cfg.dataset_name.replace(".hdf5", "")
    experiment_dir = f"./{prefix}/{cfg.benchmark_name}/{task_name}/{cfg.lifelong.algo}/" + \
            f"{cfg.policy.policy_type}_seed{cfg.seed}"

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # look for the most recent run
    experiment_id = 0
    for path in Path(experiment_dir).glob('run_*'):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split('run_')[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1

    experiment_dir += f"/run_{experiment_id:03d}"
    cfg.experiment_dir = experiment_dir
    cfg.experiment_name = "_".join(cfg.experiment_dir.split("/")[2:])
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    return True


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning

    # cfg.folder = to_absolute_path(cfg.folder) if cfg.folder is not None else get_libero_path("datasets")
    # cfg.bddl_folder = to_absolute_path(cfg.bddl_folder) if cfg.bddl_folder is not None else get_libero_path("bddl_files")
    # cfg.init_states_folder = to_absolute_path(cfg.init_states_folder) if cfg.init_states_folder is not None else get_libero_path("init_states")

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

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = 1 # benchmark.n_tasks

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    task_id = cfg.task_id

    try:
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join("./datasets",
                                            benchmark.get_task_demonstration(task_id)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len)
    except:
        print(f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}") 

    cfg.folder = os.path.join("./datasets",
                              benchmark.get_task_demonstration(task_id))
    
    cfg.dataset_name = cfg.folder.split("/")[-1]
    pp.pprint(cfg)

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    print(shape_meta)

    task_description = "" # benchmark.get_task(task_id).language
    descriptions.append(task_description)
    manip_datasets.append(task_i_dataset)

    # task_embs = get_task_embs(cfg, descriptions)
    task_embs = torch.ones((len(descriptions), 1))
    benchmark.set_task_embs(task_embs)

    datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)]

    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]


    n_tasks = n_manip_tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    # for i in range(n_tasks):

    print(f"    - Task {task_id}:")
    # print(f"        {benchmark.get_task(task_id).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    result_summary = {
    }

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    # define lifelong algorithm
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg) 
    print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    # print out the experiment folder
    print(colored(f"[info] experiment folder: {cfg.experiment_dir}", "green"))
    for i in range(n_tasks):
        print(f"[info] start training on task {i}")
        algo.train()

        with Timer() as t1:
            s_fwd, l_fwd = algo.learn_one_task(datasets[i], i, benchmark, result_summary, skip_eval_epochs=[])

        print(f'[info] train time (min) {t1.get_elapsed_time()/60.:.1f} ')
        torch.save(result_summary, os.path.join(cfg.experiment_dir, f'result.pt'))

    print("[info] finished learning\n")

    with open(os.path.join(cfg.experiment_dir, "finished.txt"), "w") as f:
        f.write("finished")    

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
