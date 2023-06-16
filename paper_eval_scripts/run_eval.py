import os
import yaml
from termcolor import colored
from easydict import EasyDict

from pathlib import Path

def main():

    checkpoint_dirs = [
        "groot_sim_results/groot/task_0/seed_seed10000",
        "groot_sim_results/groot/task_1/seed_seed10000",
        "groot_sim_results/groot/task_2/seed_seed10000",
     ]
    eval_script_name = "eval_groot.py"

    checkpoint_indices = [
        99
    ]

    camera_indices = [
        # -3, -2,
        -2, -1
        # -2, -3
        # -1,
        # 0, 1,
        # 4, 5,
    ]

    num_eval = 5
    # no_skip = False
    no_skip = True
    save_videos = "--save-videos"
    # save_videos = ""

    # checkpoint_dirs = []
    # with open("paper_results/simulation_result.yaml", "r") as f:
    #     cfg = EasyDict(yaml.safe_load(f))

    # for exp_name in cfg.keys():
    #     if exp_name not in ["VIOLA Baseline", "BC_RNN Baseline", "BC MAE Baseline"]:
    #         for exp_entry in cfg[exp_name]:
    #             checkpoint_dirs += exp_entry.run_dirs
    # print(checkpoint_dirs)

    for checkpoint_dir in checkpoint_dirs:
        eval_folder = os.path.join(checkpoint_dir, "stats_eval")
        for checkpoint_idx in checkpoint_indices:
            for camera_idx in camera_indices:

                if camera_idx <= -2:
                    if camera_idx == -2:
                        run_camera_idx = -1
                    elif camera_idx == -3:
                        run_camera_idx = -3               
                    if os.path.exists(f"{eval_folder}/stats_{run_camera_idx}_sceneTrue_{checkpoint_idx}.pt") and not no_skip:
                        print(colored(f"Skipping {checkpoint_dir} {checkpoint_idx} {run_camera_idx} because the result already exists", "red", attrs=['reverse', 'blink']))

                        continue
                    os.system(f"python paper_eval_scripts/{eval_script_name} {save_videos} --checkpoint-dir {checkpoint_dir} --checkpoint-idx {checkpoint_idx} --num-eval {num_eval} --camera-idx {run_camera_idx} --different-scene") 
                else:
                    if os.path.exists(f"{eval_folder}/stats_{camera_idx}_sceneFalse_{checkpoint_idx}.pt") and not no_skip:
                        print(colored(f"Skipping {checkpoint_dir} {checkpoint_idx} {camera_idx} because the result already exists", "red", attrs=['reverse', 'blink']))
                        continue
                    
                    os.system(f"python paper_eval_scripts/{eval_script_name} {save_videos}  --checkpoint-dir {checkpoint_dir} --checkpoint-idx {checkpoint_idx} --num-eval {num_eval} --camera-idx {camera_idx}")

if __name__ == "__main__":
    main()
