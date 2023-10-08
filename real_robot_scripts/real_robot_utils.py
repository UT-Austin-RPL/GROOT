import h5py
import numpy as np

from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys_vision.utils.calibration_utils import load_default_extrinsics, load_default_intrinsics
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info

from vos_3d_algo.o3d_modules import O3DPointCloud, convert_convention
from real_robot_scripts.groot_img_utils import ImageProcessor

class RealRobotObsProcessor():
    def __init__(self, 
                 cfg,
                 processor_name="ImageProcessor",
                 ):
        self._obs = {}
        self.cfg = cfg

        self.original_intrinsic_matrix = {}
        self.intrinsic_matrix = {}
        self.extrinsic_matrix = {}

        self.cr_interfaces = {}
        self.camera_info_dict = {}

        for camera_ref in cfg.camera_refs:
            assert_camera_ref_convention(camera_ref)
            camera_info = get_camera_info(camera_ref)
            cr_interface = CameraRedisSubInterface(camera_info, use_depth=True)
            cr_interface.start()
            self.cr_interfaces[camera_info.camera_name] = cr_interface
            self.camera_info_dict[camera_info.camera_name] = camera_info

        self.camera_names = list(self.cr_interfaces.keys())
        self.img_processor = eval(processor_name)()
        self.type_fn = lambda x: self.camera_info_dict[x].camera_type
        self.id_fn = lambda x: self.camera_info_dict[x].camera_id
        self.name_conversion_fn = lambda x: cfg.camera_name_conversion[f"{x}"]

        self.fx_fy_dict = self.img_processor.get_fx_fy_dict()

        self.original_image_size_dict = {
            "k4a": {
                0: (1280, 720),
                1: (1280, 720),
            },
            "rs": {
                0: (640, 480),
                1: (640, 480),
            }
            
        }

    def load_state(self, 
                   key, 
                   value, 
                   check_valid=True):
        if np.sum(np.abs(value)) == 0.0 and check_valid and self.last_obs_dict != {}:
            value = self.last_obs_dict[key]
        self._obs[key] = value

    def load_intrinsic_matrix(self, 
                              intrinsics=None, 
                              resize=True):

        if intrinsics is None:
            for camera_name in self.camera_names:
                intrinsic_matrix = load_default_intrinsics(self.id_fn(camera_name), self.type_fn(camera_name), image_type="color", fmt="matrix")
                self.original_intrinsic_matrix[self.name_conversion_fn(camera_name)] = intrinsic_matrix
        else:
            for camera_name in self.camera_names:                
                self.original_intrinsic_matrix[self.name_conversion_fn(camera_name)] = intrinsics[self.name_conversion_fn(camera_name)]

        if resize:
            for camera_name in self.camera_names:
                camera_type = self.type_fn(camera_name)
                camera_id = self.id_fn(camera_name)
                intrinsic_matrix = self.img_processor.resize_intrinsics(
                                        original_image_size=self.original_image_size_dict[camera_type][camera_id],
                                        intrinsic_matrix=np.array(self.original_intrinsic_matrix[self.name_conversion_fn(camera_name)]),
                                        camera_type=camera_type,
                                        img_w=self.cfg.img_w,
                                        img_h=self.cfg.img_h,
                                        fx=self.fx_fy_dict[camera_type][camera_id]["fx"],
                                        fy=self.fx_fy_dict[camera_type][camera_id]["fy"],            
                    )
                self.intrinsic_matrix[self.name_conversion_fn(camera_name)] = intrinsic_matrix
        else:
            self.intrinsic_matrix = self.original_intrinsic_matrix

    def load_extrinsic_matrix(self, extrinsics=None):
        if extrinsics is None:
            for camera_name in self.camera_names:
                extrinsic_matrix = load_default_extrinsics(self.id_fn(camera_name), self.type_fn(camera_name), fmt="matrix")
                self.extrinsic_matrix[self.name_conversion_fn(camera_name)] = extrinsic_matrix
        else:
            for camera_name in self.camera_names:
                self.extrinsic_matrix[self.name_conversion_fn(camera_name)] = extrinsics[self.name_conversion_fn(camera_name)]

    def get_extrinsic_matrix(self, key):
        if key in self.cfg.camera_name_conversion.keys():
            return self.extrinsic_matrix[self.name_conversion_fn(key)]
        else:
            return self.extrinsic_matrix[key]
    
    def get_intrinsic_matrix(self, key):
        if key in self.cfg.camera_name_conversion.keys():
            return self.intrinsic_matrix[self.name_conversion_fn(key)]
        else:
            return self.intrinsic_matrix[key]

    def get_real_robot_state(self, last_state, last_gripper_state):
        ee_states = np.array(last_state.O_T_EE)
        joint_states = np.array(last_state.q)
        gripper_states = np.array([last_gripper_state.width])

        self.load_state("ee_states", ee_states)
        self.load_state("joint_states", joint_states)
        self.load_state("gripper_states", gripper_states)

    def get_real_robot_img_obs(self):

        for camera_name in self.camera_names:
            imgs = self.cr_interfaces[camera_name].get_img()
            img_info = self.cr_interfaces[camera_name].get_img_info()
            
            color_img = imgs['color']
            depth_img = imgs['depth']

            camera_type = self.type_fn(camera_name)
            camera_id = self.id_fn(camera_name)
            resized_color_img = self.img_processor.resize_img(color_img,
                                                    camera_type=camera_type,
                                                    img_w=self.cfg.img_w,
                                                    img_h=self.cfg.img_h,
                                                    fx=self.fx_fy_dict[camera_type][camera_id]['fx'],
                                                    fy=self.fx_fy_dict[camera_type][camera_id]['fy']
                                )
            resized_depth_img = self.img_processor.resize_img(depth_img,
                                                    camera_type=camera_type,
                                                    img_w=self.cfg.img_w,
                                                    img_h=self.cfg.img_h,
                                                    fx=self.fx_fy_dict[camera_type][camera_id]['fx'],
                                                    fy=self.fx_fy_dict[camera_type][camera_id]['fy']
                                )
            self._obs[self.name_conversion_fn(camera_name) + "_rgb"] = convert_convention(resized_color_img, real_robot=True)
            self._obs[self.name_conversion_fn(camera_name) + "_depth"] = convert_convention(resized_depth_img, real_robot=True)

        return self._obs
    
    def get_original_imgs(self):

        color_imgs = []
        depth_imgs = []
        for camera_name in self.camera_names:
            imgs = self.cr_interfaces[camera_name].get_img()
            img_info = self.cr_interfaces[camera_name].get_img_info()
            
            color_img = imgs['color']
            depth_img = imgs['depth']
            color_imgs.append(color_img)
            depth_imgs.append(depth_img)
        return color_imgs, depth_imgs

    @property
    def obs(self):
        return self._obs
    

class RolloutLogger():
    def __init__(self, 
                 image_names=["agentview_rgb", "agentview_depth"]):
        self._data = {"actions": [],
                      "ee_states": [],
                      "joint_states": [],
                      "gripper_states": [],
                      "bbox": []}

        self._images = {}
        for image_name in image_names:
            self._images[image_name] = []

    def log_info(self,
                 obs_dict,
                 action):
        self._data["actions"].append(action)

        self._data["ee_states"].append(obs_dict["ee_states"])
        self._data["joint_states"].append(obs_dict["joint_states"])
        self._data["gripper_states"].append(obs_dict["gripper_states"])

        for image_name in self._images.keys():
            self._images[image_name].append(obs_dict[image_name])

    def save(self, folder_name):

        with h5py.File(f"{folder_name}/rollout.hdf5", "w") as rollout_file:
            grp = rollout_file.create_group("data")
            grp.create_dataset("action", data=np.stack(self._data["actions"], axis=0))
            grp.create_dataset("ee_states", data=np.stack(self._data["ee_states"], axis=0))
            grp.create_dataset("joint_states", data=np.stack(self._data["joint_states"], axis=0))
            grp.create_dataset("gripper_states", data=np.stack(self._data["gripper_states"], axis=0))
            grp.create_dataset("agentview_rgb", data=np.stack(self._images["agentview_rgb"], axis=0))
            grp.create_dataset("agentview_depth", data=np.stack(self._images["agentview_depth"], axis=0))
            