import torch
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from libero.lifelong.utils import *


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """ 
        Prepare the tensor observations as input for the algorithm.
    """

    data = {
        "obs": {},
        "task_emb": task_emb
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for obs_name in all_obs_keys:
        if obs_name in cfg.data.obs_key_mapping:
            mapped_name = cfg.data.obs_key_mapping[obs_name]
        else:
            mapped_name = obs_name
        if "neighborhood" in mapped_name or "centers" in mapped_name:
            continue
        data["obs"][obs_name] = torch.from_numpy(ObsUtils.process_obs(
            obs[mapped_name],
            obs_key=obs_name)).float().unsqueeze(0)

    data = TensorUtils.map_tensor(data,
                                  lambda x: safe_device(x, device=cfg.device))
    return data


def raw_real_obs_to_tensor_obs(obs, task_emb, cfg):
    """ 
        Prepare the tensor observations as input for the algorithm.
    """

    data = {
        "obs": {},
        "task_emb": task_emb
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for obs_name in all_obs_keys:
        mapped_name = obs_name
        if "neighborhood" in mapped_name or "centers" in mapped_name:
            continue
        data["obs"][obs_name] = torch.from_numpy(ObsUtils.process_obs(
            obs[mapped_name],
            obs_key=obs_name)).float().unsqueeze(0)

    data = TensorUtils.map_tensor(data,
                                  lambda x: safe_device(x, device=cfg.device))
    return data
