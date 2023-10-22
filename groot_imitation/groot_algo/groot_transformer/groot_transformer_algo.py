import copy
import time
import gc
import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.models import *
from libero.lifelong.utils import *

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv 
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.utils.time_utils import Timer



def evaluate_one_task_success(cfg, algo, task, task_emb, task_id,
                              sim_states=None,
                              task_str="",
                              camera_idx=-1):
    """
        Evaluate a single task's success rate
        sim_states: if not None, will keep track of all simulated states during
                    evaluation, mainly for visualization and debugging purpose
        task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet": # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
            # "camera_idx": camera_idx,
            "camera_depths": True,
        }


        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(cfg.init_states_folder,
                                        task.problem_folder,
                                        task.init_states_file)
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(cfg.eval.n_eval):
            env.reset()
            init_states_ = init_states[i]

            done = False
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)

            for _ in range(5):
                obs, _, _, _ = env.step([0.] * 7)

            if task_str != "":
                sim_state = env.get_sim_state()
                if sim_states is not None:
                    sim_states[task_str].append(sim_state)

            while steps < cfg.eval.max_steps:
                steps += 1

                # This sentence needs to be changed.
                # The conversion should also contain the XMem model, which will directly give object centric information over the masks.
                raise NotImplementedError
                data = raw_obs_to_tensor_obs(obs, task_emb, cfg, env.sim)

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # record the sim states for replay purpose
                if task_str != "":
                    sim_state = env.get_sim_state()
                    if sim_states is not None:
                        sim_states[0].append(sim_state)


                if done: break

            num_success += int(done)

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate

class GROOTSingleTask(Sequential):
    """
        The sequential BC baseline.
    """
    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)

    def start_task(self, task):
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
                self.policy.parameters(),
                **self.cfg.train.optimizer.kwargs)

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            if self.cfg.train.scheduler.name == "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts":
                self.scheduler = eval(self.cfg.train.scheduler.name)(
                        self.optimizer,
                        **self.cfg.train.scheduler.kwargs)
            else:
                self.scheduler = eval(self.cfg.train.scheduler.name)(
                        self.optimizer,
                        T_max=self.cfg.train.n_epochs,
                        **self.cfg.train.scheduler.kwargs)


    def learn_one_task(self, dataset, task_id, benchmark, result_summary, skip_eval_epochs=[]):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id*gsz, (task_id+1)*gsz))

        model_checkpoint_name = os.path.join(self.experiment_dir,
                                             f"task{task_id}_model.pth")
        
        train_dataloader = DataLoader(dataset,
                                      batch_size=self.cfg.train.batch_size,
                                      num_workers=self.cfg.train.num_workers,
                                      shuffle=True)

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict() # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        # task = benchmark.get_task(task_id)
        # task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(1, self.cfg.train.n_epochs+1):

            t0 = time.time()
            if epoch > 0: # update 
                self.policy.train()
                training_loss = 0.
                for (idx, data) in enumerate(tqdm(train_dataloader)):
                    loss = self.observe(data)
                    if self.cfg.train.scheduler.name == "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts" and self.scheduler is not None:
                        self.scheduler.step(epoch + idx / len(train_dataloader))

                    training_loss += loss
                training_loss /= len(train_dataloader)
                
            else: # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}")

            # TODO(@Bo) Find a better solution, it is caused by the num_workers in dataloader
            time.sleep(0.1)

            if epoch % self.cfg.eval.eval_every == 0 and epoch not in skip_eval_epochs: # evaluate BC loss
                self.policy.eval()
                losses.append(training_loss)

                t0 = time.time()

                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = result_summary[task_str] if self.cfg.eval.save_sim_states else None
                success_rate = 0

                successes.append(success_rate)

                torch_save_model(self.policy, model_checkpoint_name.replace(".pth", f"_{epoch}.pth"), cfg=self.cfg)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0

                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)

                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} " + \
                        f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}", flush=True)

            if self.cfg.train.scheduler.name == "torch.optim.lr_scheduler.CosineAnnealing" and self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        self.end_task(dataset, task_id, benchmark)

        # return the metrics regarding forward transfer
        loss_at_best_succ = losses[idx_at_best_succ]
        success_at_best_succ = successes[idx_at_best_succ]

        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(self.experiment_dir,
                                             f"task{task_id}_auc.log")
        torch.save({
            "success": successes,
            "loss": losses,}, auc_checkpoint_name)

        losses[idx_at_best_succ:] = loss_at_best_succ 
        successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter