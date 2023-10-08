# After dataset creation and interactive UI, run this script
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

import init_path
from vos_3d_algo.misc_utils import get_annotation_path

def run_commands(commandline):
    commands = commandline.split(" ")
    try:
        subprocess.run(commands, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running", colored(" ".join(commands), "yellow"))
        print(e)
        exit()
    return commands

@hydra.main(config_path="./data_configs", config_name="config", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    assert(os.path.exists(cfg.dataset_name)), f"Dataset {cfg.dataset_name} does not exist"
    assert(cfg.mode in ["single_instance", "multiple_instances"])


    dataset_name = cfg.dataset_name

    is_real_dataset = False
    with h5py.File(dataset_name, "r") as f:
        num_demos = len(f["data"].keys())
        

        if "real" in f["data"].attrs.keys() and f["data"].attrs["real"]:
            is_real_dataset = True

    with h5py.File(dataset_name, "r+") as f:
        f["data"].attrs["yaml_config"] = json.dumps(cfg)

    print("Successully updated the dataset attributes")

    if is_real_dataset:
        print(colored(pyfiglet.figlet_format("Real Datasets", font="standard"), "green", attrs=['reverse', 'blink']))

    annotation_folder = get_annotation_path(dataset_name)
    if cfg.mode == "single_instance":

        # generate tracking masks
        print(pyfiglet.figlet_format("Tracking Masks", font="standard"))
        if cfg.save_video:
            arguments = "--save-video"
            if is_real_dataset:
                arguments += " --real"
            commandline = f"python data_preprocessing/preprocess_dataset.py --dataset {dataset_name} {arguments}"
            # run_commands(commandline)
            # assert if the file exists to further verify if the script is successful.
            assert(os.path.exists(os.path.join(annotation_folder, "masks.hdf5"))), \
                colored("Error creating tracking masks", "red")

    # create point clouds
    print(pyfiglet.figlet_format("Generating Point Clouds", font="standard"))
    if is_real_dataset:
        arguments = "--real"
    if cfg.datasets.erode:
        arguments += " --erode"
    commandline = f"python data_preprocessing/pcd_dataset_generation.py --dataset {dataset_name} {arguments}"

    # run_commands(commandline)
    # assert(os.path.exists(os.path.join(annotation_folder, "pcd.hdf5"))), \
    #     colored("Error creating point clouds", "red")


    # start augmentation on original datasets
    print(pyfiglet.figlet_format("Expanding The Demo", font="standard"))
    aug_dataset_name = dataset_name.replace("demo.hdf5", "aug_demo.hdf5")

    if cfg.datasets.aug_rotation != 0:
        scale = 3
    else:
        scale = 2    

    commandline = f"python data_preprocessing/aug_demo.py --dataset {dataset_name} --new-dataset {aug_dataset_name} --num-demos {num_demos} --scale {scale}"

    run_commands(commandline)

    aug_annotation_folder = get_annotation_path(aug_dataset_name)
    os.makedirs(aug_annotation_folder, exist_ok=True)
    assert(os.path.exists(aug_dataset_name)), \
        colored("Error augmenting dataset", "red")
    
    print(pyfiglet.figlet_format("Trasnferring annotations", font="standard"))
    shutil.copy2(os.path.join(annotation_folder, "frame.jpg"), os.path.join(aug_annotation_folder, "frame.jpg"))
    shutil.copy2(os.path.join(annotation_folder, "frame_annotation.png"), os.path.join(aug_annotation_folder, "frame_annotation.png"))
                                                                    

    # start augmenting point clouds
    print(pyfiglet.figlet_format("Augmenting Point Clouds", font="standard"))
    print(f"rotating by {cfg.datasets.aug_rotation} degrees")
    arguments = f"--rotation {cfg.datasets.aug_rotation} --num-demos {num_demos}"
    if is_real_dataset:
        arguments += " --real"

    if cfg.datasets.erode:
        arguments += " --erode"
    if is_real_dataset:
        commandline = f"python data_preprocessing/aug_demo_generation_real.py --dataset {dataset_name} {arguments}"
    else:
        commandline = f"python data_preprocessing/aug_demo_generation_default.py --dataset {dataset_name} {arguments}"

    run_commands(commandline)

    aug_pcd_dataset = os.path.join(annotation_folder, "aug_pcd.hdf5")
    assert(os.path.exists(aug_pcd_dataset)), \
        colored("Error augmenting point clouds", "red")
    
    # start grouping point clouds
    print(pyfiglet.figlet_format("Merging Point Cloud to Aug Dataset", font="standard"))
    commandline = f"kaede.extra.edit_demo --base-dataset {aug_dataset_name} --additional-dataset {aug_pcd_dataset} --merge-mode"
    run_commands(commandline)
    
    # start grouping point clouds
    print(pyfiglet.figlet_format("Grouping Point Clouds", font="standard"))
    num_group = cfg.datasets.num_group
    group_size = cfg.datasets.group_size
    arguments = f"--num-group {num_group} --group-size {group_size}"
    if is_real_dataset:
        arguments += " --real"
    commandline = f"python data_preprocessing/pcd_grouping_preprocess.py --dataset {aug_dataset_name} {arguments}"

    run_commands(commandline)

    assert(os.path.exists(os.path.join(aug_annotation_folder, "grouped_pcd.hdf5"))), \
        colored("Error grouping point clouds", "red")

    grouped_pcd_file = os.path.join(aug_annotation_folder, "grouped_pcd.hdf5")
    commandline = f"kaede.extra.edit_demo --base-dataset {aug_dataset_name} --additional-dataset {grouped_pcd_file} --merge-mode"
    run_commands(commandline)

    # Put first frame annotation into the dataset
    print(pyfiglet.figlet_format("Putting First Frame Annotation into Dataset", font="standard"))
    commandline = f"python data_preprocessing/put_first_frame_annotation_in_dataset.py --dataset {dataset_name}"
    run_commands(commandline)
    commandline = f"python data_preprocessing/put_first_frame_annotation_in_dataset.py --dataset {aug_dataset_name}"
    run_commands(commandline)


    # # print the dataset info
    # commandline = f"kaede_utils.kaede.get_dataset_info --dataset {dataset_name}"

    info = {
        "dataset_name": dataset_name,
        "aug_dataset_name": aug_dataset_name,
        "mode": cfg.mode,
        "save_video": cfg.save_video,
        "pcd: ": os.path.join(annotation_folder, "pcd.hdf5"),
        "aug_pcd: ": os.path.join(annotation_folder, "aug_pcd.hdf5"),
        "grouped_pcd: ": os.path.join(annotation_folder, "grouped_pcd.hdf5"),
        "masks: ": os.path.join(annotation_folder, "masks.hdf5")
    }

    with open(os.path.join(annotation_folder, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=4)
    for key, value in info.items():
        print(key, ": ",  value)

if __name__ == "__main__":
    main()