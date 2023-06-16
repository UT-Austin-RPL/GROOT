import os
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

import copy
import gc
import numpy as np
import os
import robosuite.utils.transform_utils as T
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.envs.venv import SubprocVectorEnv 
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark

from kaede_utils.robosuite_utils.xml_utils import postprocess_model_xml, get_camera_info_from_xml

def rotate_real_camera(camera_pose, angle=-10, point=[0, 0, 0], axis=[0, 0, 1]):   
    rad = np.pi * angle / 180.0

    homo_rot_z = T.make_pose(np.array([0., 0., 0.]), T.quat2mat(T.axisangle2quat(np.array(axis) * rad)))

    new_camera_pose = homo_rot_z @ camera_pose
    new_camera_pose[:3, 3] = new_camera_pose[:3, 3] + np.array(point)

    return new_camera_pose

def rotate_camera(camera_pos, camera_quat, angle=-10, point=[0, 0, 0], axis=[0, 0, 1]):   
    camera_rot = T.quat2mat(T.convert_quat(camera_quat, to="xyzw"))
    rad = np.pi * angle / 180.0

    homo_rot_z = T.make_pose(np.array([0., 0., 0.]), T.quat2mat(T.axisangle2quat(np.array(axis) * rad)))
    camera_pose = np.zeros((4, 4))
    camera_pose[:3, :3] = camera_rot
    camera_pose[:3, 3] = camera_pos

    camera_pose = homo_rot_z @ camera_pose


    # Update camera pose
    pos, quat = camera_pose[:3, 3] + np.array(point), T.mat2quat(camera_pose[:3, :3])
    quat = T.convert_quat(quat, to="wxyz")
    return pos, quat

class ViewChangingOffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """
    def __init__(self, camera_idx=-1, training_mode=False, **kwargs):
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True

        self.camera_idx = camera_idx

        if not training_mode:
            self.camera_rotation_config_list = [
                # rotate around z axis
                (-80, [0, 0, 1], [0., 0.0, 0.0]),
                (-40, [0, 0, 1], [0., 0.0, 0.0]),
                (-20, [0, 0, 1], [0., 0.0, 0.0]),
                (20, [0, 0, 1], [0., 0.0, 0.0]),
                (40, [0, 0, 1], [0., 0.0, 0.0]),
                (80, [0, 0, 1], [0., 0.0, 0.0]),
                (0, [0, 0, 1], [-0.05, 0.0, 0.0]),
                (0, [0, 0, 1], [-0.1, 0.0, 0.0]),
                (0, [0, 0, 1], [0.05, 0.0, 0.0]),
                (0, [0, 0, 1], [0.1, 0.0, 0.0]),
                (0, [0, 0, 1], [0.0, 0.0, 0.05]),
                (0, [0, 0, 1], [0.0, 0.0, -0.05]),
            ]
        else:
            self.camera_rotation_config_list = [
                # rotate around z axis
                (-30, [0, 0, 1], [0., 0.0, 0.0]),
                (-10, [0, 0, 1], [0., 0.0, 0.0]),
                (-5, [0, 0, 1], [0., 0.0, 0.0]),
                (0,   [0, 0, 1], [0., 0.0, 0.0,]),
                (5, [0, 0, 1], [0., 0.0, 0.0]),
                (10, [0, 0, 1], [0., 0.0, 0.0]),
                (30, [0, 0, 1], [0., 0.0, 0.0]),
            ]

        super().__init__(**kwargs)

    def set_camera_idx(self, camera_idx):
        self.camera_idx = camera_idx
    
    def reset(self):
        obs = self.env.reset()
        
        model_xml = self.env.sim.model.get_xml()
        canonical_camera_pos, canonical_camera_quat = get_camera_info_from_xml(model_xml, "agentview")

        if self.camera_idx >= 0:
            angle, axis, point = self.camera_rotation_config_list[self.camera_idx]
            camera_pos, camera_quat = rotate_camera(canonical_camera_pos, canonical_camera_quat, angle=angle, axis=axis, point=point)

            model_xml = self.env.sim.model.get_xml()
            camera_dict = {"agentview": {"pos": camera_pos, "quat": camera_quat}}
            model_xml = postprocess_model_xml(model_xml, camera_dict)
            initial_mjstate = self.env.sim.get_state().flatten()            
            self.env.reset_from_xml_string(model_xml)
            obs = self.set_init_state(initial_mjstate)

        return obs