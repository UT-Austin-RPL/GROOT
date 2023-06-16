"""In this script, we will launch a single training experiment."""

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
                            torch_load_model, create_experiment_dir, get_task_embs)
from libero.libero.utils.time_utils import Timer

import init_path
from vos_3d_algo.vos_3d_transformer import VOS3DSingleTask
from vos_3d_algo.baselines import BCRNNRGBDPolicy
from robomimic.utils.obs_utils import Modality
from vos_3d_algo import PcdModality, NormalizedPcdModality, VOS_3D_Benchmark, WristDepthModality, GroupedPcdModality, VOS_3D_Ablation_Augmentation_Benchmark, VOS_3D_Ablation_Grouping_Benchmark


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning

    cfg.folder = to_absolute_path(cfg.folder) if cfg.folder is not None else get_libero_path("datasets")
    cfg.bddl_folder = to_absolute_path(cfg.bddl_folder) if cfg.bddl_folder is not None else get_libero_path("bddl_files")
    cfg.init_states_folder = to_absolute_path(cfg.init_states_folder) if cfg.init_states_folder is not None else get_libero_path("init_states")


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

    print(shape_meta)

    task_description = benchmark.get_task(task_id).language
    descriptions.append(task_description)
    manip_datasets.append(task_i_dataset)

    # task_embs = get_task_embs(cfg, descriptions)
    task_embs = torch.ones((len(descriptions), 1))
    benchmark.set_task_embs(task_embs)

    datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)]

    # # Temporarily test pcd data loading
    
    # train_dataloader = torch.utils.data.DataLoader(datasets[0],
    #                                                 batch_size=cfg.train.batch_size,
    #                                                 num_workers=cfg.train.num_workers,
    #                                                 shuffle=True)
    # for data in train_dataloader:
    #     import pdb; pdb.set_trace()

    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]


    n_tasks = n_manip_tasks # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    # for i in range(n_tasks):

    print(f"    - Task {task_id}:")
    print(f"        {benchmark.get_task(task_id).language}")
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
    print(colored(f"[info] experiment folder: {cfg.experiment_dir}", "green"))

if __name__ == "__main__":
    main()
