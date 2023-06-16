import os
dirname = os.path.dirname(__file__)
xmem_checkpoint_folder = os.path.join(dirname, "./xmem_checkpoints")

import numpy as np
from robomimic.utils.obs_utils import Modality, process_frame
from libero.libero.benchmark import get_benchmark, Benchmark, task_maps, register_benchmark


class GroupedPcdModality(Modality):
    name = "grouped_pcd"
    @classmethod
    def _default_obs_processor(cls, obs):
        # We add a channel dimension and normalize them to be in range [-1, 1]
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # We do the reverse
        return obs

class PcdModality(Modality):
    name = "pcd"
    @classmethod
    def _default_obs_processor(cls, obs):
        # We add a channel dimension and normalize them to be in range [-1, 1]
        start_dims = np.arange(len(obs.shape) - 3).tolist()
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # We do the reverse
        start_dims = np.arange(len(obs.shape) - 3).tolist()
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])
        
# For robosuite simulation only for now
def normalize_kitchen3_point_cloud(obs):
    max_array = np.array([0.231, 0.794, 1.444], dtype=np.float32)
    min_array = np.array([-1.72, -0.99, 0.36], dtype=np.float32)
    return (obs - min_array) / (max_array - min_array)

def normalize_real_robot_point_cloud(obs):
    max_array = np.array([0.69943695, 0.5, 0.45784091], dtype=np.float32)
    min_array = np.array([0.0, -0.5, 0.0], dtype=np.float32)
    # print("You need to double check this part")
    return (obs - min_array) / (max_array - min_array)

def normalize_real_robot_point_cloud_obs_processor(obs):
        start_dims = np.arange(len(obs.shape) - 3).tolist()

        obs = normalize_real_robot_point_cloud(obs)
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])
    

def normalize_pcd(obs, real_robot=False):
    if real_robot:
        return normalize_real_robot_point_cloud(obs)
    else:
        return normalize_kitchen3_point_cloud(obs)
        
class NormalizedPcdModality(Modality):
    name = "normalized_pcd"
    @classmethod
    def _default_obs_processor(cls, obs):
        # We add a channel dimension and normalize them to be in range [-1, 1]
        start_dims = np.arange(len(obs.shape) - 3).tolist()
        obs = normalize_kitchen3_point_cloud(obs)
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # We do the reverse
        start_dims = np.arange(len(obs.shape) - 3).tolist()
        if isinstance(obs, np.ndarray):
            return obs.transpose(start_dims + [-3, -1, -2])
        else:
            return obs.permute(start_dims + [-3, -1, -2])        

class WristDepthModality(Modality):
    """
    Modality for depth observations
    """
    name = "wrist_depth"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given depth fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 1] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            processed_obs (np.array or torch.Tensor): processed depth
        """
        max_depth = 100
        obs[obs > max_depth] = max_depth
        return process_frame(frame=obs, channel_dim=1, scale=max_depth)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given depth prepared for network input, prepare for saving to dataset.
        Inverse of @process_depth.

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): depth passed through
                inverse operation of @process_depth
        """
        return TU.to_uint8(unprocess_frame(frame=obs, channel_dim=1, scale=1.))


class RealDepthModality(Modality):
    """
    Modality for depth observations
    """
    name = "real_depth"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given depth fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 1] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            processed_obs (np.array or torch.Tensor): processed depth
        """
        # check if obs is a tensor or numpy
        if isinstance(obs, np.ndarray):
            obs = obs[..., np.newaxis]
        elif isinstance(obs, torch.Tensor):
            obs = obs.unsqueeze(-1)
        max_depth = 1000
        obs[obs > max_depth] = max_depth            
        return process_frame(frame=obs, channel_dim=1, scale=max_depth)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given depth prepared for network input, prepare for saving to dataset.
        Inverse of @process_depth.

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): depth passed through
                inverse operation of @process_depth
        """
        return TU.to_uint8(unprocess_frame(frame=obs, channel_dim=1, scale=1.))

def custom_real_wrist_dept_process(obs):
    max_depth = 100
    print("You wanna double check this part")
    import pdb; pdb.set_trace()
    obs[obs > max_depth] = max_depth
    return process_frame(frame=obs, channel_dim=1, scale=max_depth)

def toggle_data_modality_processing(real_robot=False):
    if real_robot:
        NormalizedPcdModality.set_obs_processor(normalize_real_robot_point_cloud_obs_processor)
        WristDepthModality.set_obs_processor(custom_real_wrist_dept_process)

#  ____                  _                          _        
# | __ )  ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____ 
# |  _ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
# | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
# |____/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/

@register_benchmark
class VOS_3D_Simulation_Benchmark(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "VOS_3D_Simulation_Benchmark"
        self.task_orders = [[0, 1, 2, 3, 4, 5, 6]]
        self._make_benchmark()

    def _make_benchmark(self):
        tasks = list(task_maps["libero_10"].values())

        self.dataset_name = [
            "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_aug",
            "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_aug",
            "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_aug",
        ]
        tasks = [
            task_maps["libero_90"]["KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"],
            task_maps["libero_90"]["KITCHEN_SCENE3_put_the_frying_pan_on_the_stove"],
            task_maps["libero_90"]["KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug"],
        ]
        print(f"[info] using task orders {self.task_orders[self.task_order_index]}")
        self.tasks = [tasks[i] for i in self.task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_task_demonstration(self, i):
        assert 0 <= i and i < self.n_tasks, \
                f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"libero_datasets/{self.dataset_name[i]}_demo.hdf5"
        return demo_path    
    

@register_benchmark
class VOS_3D_Simulation_RGBD_Benchmark(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "VOS_3D_Simulation_Benchmark"
        self.task_orders = [[0, 1, 2, 3, 4, 5]]
        self._make_benchmark()

    def _make_benchmark(self):
        tasks = list(task_maps["libero_10"].values())

        self.dataset_name = [
            "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
            "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
            "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        ]
        tasks = [
            task_maps["libero_90"]["KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"],
            task_maps["libero_90"]["KITCHEN_SCENE3_put_the_frying_pan_on_the_stove"],
            task_maps["libero_90"]["KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug"],
        ]
        print(f"[info] using task orders {self.task_orders[self.task_order_index]}")
        self.tasks = [tasks[i] for i in self.task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_task_demonstration(self, i):
        assert 0 <= i and i < self.n_tasks, \
                f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"libero_datasets/{self.dataset_name[i]}_demo.hdf5"
        return demo_path


@register_benchmark
class VOS_3D_Real_Robot_Benchmark(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "VOS_3D_Benchmark"
        self.task_orders = [[0, 1, 2, 3, 4]]
        self._make_benchmark()

    def _make_benchmark(self):
        tasks = list(task_maps["libero_10"].values())

        self.dataset_name = [
            "pick_place_cup_aug",
            "stamp_paper_aug",
            "take_the_mug_aug",
            "pick_place_mug_aug",
            "roller_stamp_aug"
        ]

        print(f"[info] using task orders {self.task_orders[self.task_order_index]}")
        # self.tasks = [tasks[i] for i in self.task_orders[self.task_order_index]]
        self.n_tasks = len(self.dataset_name)

    def get_task_demonstration(self, i):
        assert 0 <= i and i < self.n_tasks, \
                f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"real_datasets/{self.dataset_name[i]}_demo.hdf5"

        return demo_path
    


@register_benchmark
class VOS_3D_Real_Robot_RGBD_Benchmark(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "VOS_3D_Benchmark"
        self.task_orders = [[0, 1, 2, 3, 4]]
        self._make_benchmark()

    def _make_benchmark(self):
        tasks = list(task_maps["libero_10"].values())

        # Remap the task ids in the original codebase to reduce the id by 3
        self.dataset_name = [
            "pick_place_cup",
            "stamp_paper",
            "take_the_mug",
            "pick_place_mug",
            "roller_stamp"
        ]

        print(f"[info] using task orders {self.task_orders[self.task_order_index]}")
        # self.tasks = [tasks[i] for i in self.task_orders[self.task_order_index]]
        self.n_tasks = len(self.dataset_name)

    def get_task_demonstration(self, i):
        assert 0 <= i and i < self.n_tasks, \
                f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"real_datasets/{self.dataset_name[i]}_demo.hdf5"

        return demo_path
    


