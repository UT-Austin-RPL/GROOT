import os



def main():

    checkpoint_dirs = [
        "experiments_paper/bc_mae_baselines/exp_task_0_seed10000",
        "experiments_paper/bc_mae_baselines/exp_task_2_seed10000",
        "experiments_paper/bc_mae_baselines/exp_task_3_seed10000",
        "experiments_paper/bc_rnn_baselines/bc_rnn_simulation_task_2_seed10000",
        "experiments_paper/bc_rnn_baselines/bc_rnn_simulation_task_3_seed10000",
        "experiments_paper/bc_rnn_baselines/bc_rnn_simulation_task_0_seed10000",        
    ]
    eval_script_name = "eval_rnn_mae.py"

    checkpoint_indices = [
        99
    ]

    camera_indices = [
        -2, -3
        -1,
        0, 1,
        4, 5,
    ]

    num_eval = 5
    save_videos = "--save-videos"

    for checkpoint_dir in checkpoint_dirs:
        eval_folder = os.path.join(checkpoint_dir, "stats_eval")
        for checkpoint_idx in checkpoint_indices:
            for camera_idx in camera_indices:
                if camera_idx <= -2:
                    if camera_idx == -2:
                        run_camera_idx = -1
                    else:
                        run_camera_idx = camera_idx                    
                    if os.path.exists(f"{eval_folder}/stats_{run_camera_idx}_sceneTrue_{checkpoint_idx}.pt"):
                        print(f"Skipping {checkpoint_dir} {checkpoint_idx} {run_camera_idx} because the result already exists")
                        continue
                    os.system(f"python paper_eval_scripts/{eval_script_name} {save_videos} --checkpoint-dir {checkpoint_dir} --checkpoint-idx {checkpoint_idx} --num-eval {num_eval} --camera-idx {run_camera_idx} --different-scene")
                else:
                    if os.path.exists(f"{eval_folder}/stats_{camera_idx}_sceneFalse_{checkpoint_idx}.pt"):
                        print(f"Skipping {checkpoint_dir} {checkpoint_idx} {camera_idx} because the result already exists")
                        continue
                    
                    os.system(f"python paper_eval_scripts/{eval_script_name} {save_videos}  --checkpoint-dir {checkpoint_dir} --checkpoint-idx {checkpoint_idx} --num-eval {num_eval} --camera-idx {camera_idx}")

if __name__ == "__main__":
    main()
