import argparse
import h5py
import os
import json
import subprocess

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import yaml
import pyfiglet


from omegaconf import OmegaConf
from easydict import EasyDict
from termcolor import colored
import shutil
import pyfiglet

import init_path
from groot_imitation.groot_algo.misc_utils import get_annotation_path, edit_h5py_datasets, Timer
from groot_imitation.groot_algo.dataset_preprocessing.vos_annotation import dataset_vos_annotation
from groot_imitation.groot_algo.dataset_preprocessing.pcd_generation import object_pcd_generation, pcd_grouping
from groot_imitation.groot_algo.xmem_tracker import XMemTracker


@hydra.main(config_path="./dataset_configs", config_name="config", version_base=None)
def main(hydra_cfg):

    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    print(cfg)
    device = "cuda:0"
    assert(cfg.dataset_path is not None), "Please specify the dataset name"

    original_dataset_name = cfg.dataset_path

    with Timer(verbose=True) as timer:
        annotation_folder = get_annotation_path(original_dataset_name)
        print(annotation_folder)

        xmem_tracker = XMemTracker(xmem_checkpoint=f'third_party/xmem_checkpoints/XMem.pth', device=device)
        xmem_tracker.clear_memory()

        mask_dataset_name = os.path.join(annotation_folder, "masks.hdf5")


        # generate masks using xmem and directly do the tracking 
        if cfg.vos_annotation:
            print(pyfiglet.figlet_format("XMem Annotation", font="standard"))
            dataset_vos_annotation(
                cfg,
                dataset_name=original_dataset_name,
                mask_dataset_name=mask_dataset_name, 
                xmem_tracker=xmem_tracker,
                annotation_folder=annotation_folder,
                save_video=True,
                verbose=False,
                )
        
        # Generate object-centric point clouds for training

        pcd_dataset_name = os.path.join(annotation_folder, "pcd.hdf5")
        pcd_grouped_dataset_name = os.path.join(annotation_folder, "pcd_grouped.hdf5")

        if cfg.object_pcd:
            print(pyfiglet.figlet_format("Point Cloud Generation", font="standard"))
            if cfg.pcd_aug:
                print("Doing augmentation here")
            aug_demo_mapping = object_pcd_generation(
                cfg,
                dataset_name=original_dataset_name,
                mask_dataset_name=mask_dataset_name, 
                pcd_dataset_name=pcd_dataset_name,
                annotation_folder=annotation_folder,
                )
        else:
            # we need to retrieve from dataset about aug_demo_mapping
            with h5py.File(pcd_dataset_name, 'r') as pcd_dataset:
                aug_demo_mapping = json.loads(pcd_dataset.attrs["aug_demo_mapping"])

        if cfg.pcd_grouping:
            print(pyfiglet.figlet_format("Point Cloud Grouping", font="standard"))
            pcd_grouping(
                cfg,
                pcd_dataset_name=pcd_dataset_name,
                pcd_grouped_dataset_name=pcd_grouped_dataset_name,
            )

        print(pyfiglet.figlet_format("Merging", font="standard"))
        # Generate the final training dataset
        training_set_name = original_dataset_name.replace(".hdf5", "_training.hdf5")
        with h5py.File(original_dataset_name, 'r') as original_dataset, \
            h5py.File(pcd_grouped_dataset_name, 'r') as pcd_grouped_dataset, \
            h5py.File(training_set_name, 'w') as training_dataset:
            training_dataset.create_group("data")
            for key in original_dataset["data"].attrs.keys():
                training_dataset["data"].attrs[key] = original_dataset["data"].attrs[key]

            for demo in pcd_grouped_dataset["data"].keys():
                if demo not in aug_demo_mapping:
                    source_demo = demo
                else:
                    source_demo = aug_demo_mapping[demo]

                training_dataset.create_group(f"data/{demo}/obs")

                for key in original_dataset[f"data/{source_demo}"].attrs.keys():
                    training_dataset[f"data/{demo}"].attrs[key] = original_dataset[f"data/{source_demo}"].attrs[key]

                for key in ["joint_states", "gripper_states", "ee_states", "agentview_extrinsics"]:
                    original_dataset.copy(f"data/{source_demo}/obs/{key}", training_dataset[f"data/{demo}/obs"], name=key)

                original_dataset.copy(f"data/{source_demo}/actions", training_dataset[f"data/{demo}"], name="actions")

        edit_h5py_datasets(training_set_name, pcd_grouped_dataset_name, mode="merge")

        if cfg.delete_intermediate_files:
            print(pyfiglet.figlet_format("Delete Intermediate Files", font="standard"))
            os.remove(pcd_dataset_name)
            os.remove(pcd_grouped_dataset_name)

if __name__ == "__main__":
    main()
