
import json
import os
import time
import argparse


import numpy as np
from pathlib import Path

from rpl_vision_utils.networking.camera_redis_interface import \
    CameraRedisSubInterface

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from rpl_vision_utils.utils.calibration_utils import load_default_extrinsics, load_default_intrinsics

from kaede_utils.networking_utils.redis_utils import RedisInteractiveInterface

logger = get_deoxys_example_logger()


class DeoxysDataCollection():
    def __init__(self, 
                 observation_cfg,
                 folder, 
                 interface_cfg="charmander.yml", 
                 controller_cfg_name="osc-pose-controller.yml", 
                 controller_type="OSC_POSE", 
                 max_steps=2000,
                 vendor_id=9583, 
                 product_id=50734,
                 ):
        self.folder = Path(folder)
        self.interface_cfg = interface_cfg
        self.controller_cfg_name = controller_cfg_name
        self.controller_type = controller_type
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.observation_cfg = observation_cfg
        self.camera_ids = observation_cfg.camera_ids

        self.folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {self.folder}")
        self.max_steps = max_steps
        
        self.tmp_folder = None
        self.tmp_data = None

        self.controller_cfg = {}

        self.robot_interface = None

        self.intrinsics = {}
        self.extrinsics = {}

    def collect_data(self):
        experiment_id = 0

        for path in self.folder.glob("run*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        folder = str(self.folder / f"run{experiment_id:03d}")

        device = SpaceMouse(vendor_id=self.vendor_id, product_id=self.product_id)
        device.start_control()

        self.robot_interface = FrankaInterface(os.path.join(config_root, self.interface_cfg))

        cr_interfaces = {}
        for camera_id in self.camera_ids:
            cr_interface = CameraRedisSubInterface(camera_id=camera_id)
            cr_interface.start()
            cr_interfaces[camera_id] = cr_interface

        controller_cfg = YamlConfig(
            os.path.join(config_root, self.controller_cfg_name)
        ).as_easydict()

        data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}
        for camera_id in self.camera_ids:
            data[f"camera_{camera_id}"] = []
        i = 0
        start = False

        previous_state_dict = None

        time.sleep(2)

        # Record intrinsics and extrinsics
        # We wil save both of them as a single file
        # Unlike in robosuite, the value of eye-in-hand extrinsic is still fixed (with respect to the end effector)
        # In the create dataset step, we need to convert this value to actual eye in hand camera to the base. 

        type_fn = lambda x: self.observation_cfg.camera_types[f"camera_{x}"]
        for camera_id in self.camera_ids:
            camera_type = type_fn(camera_id)
            extrinsics = load_default_extrinsics(camera_id, camera_type, calibration_method="tsai", fmt="dict")
            intrinsics = load_default_intrinsics(camera_id, camera_type, fmt="dict")

            self.intrinsics[camera_id] = intrinsics
            self.extrinsics[camera_id] = extrinsics

        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            action, grasp = input2action(
                device=device,
                controller_type=self.controller_type,
            )
            if action is None:
                break

            if self.controller_type == "OSC_YAW":
                action[3:5] = 0.0
            elif self.controller_type == "OSC_POSITION":
                action[3:6] = 0.0

            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )

            if len(self.robot_interface._state_buffer) == 0:
                continue
            last_state = self.robot_interface._state_buffer[-1]
            last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
            if np.linalg.norm(action[:-1]) < 1e-3 and not start:
                continue

            start = True

            data["action"].append(action)

            state_dict = {
                "ee_states": np.array(last_state.O_T_EE),
                "joint_states": np.array(last_state.q),
                "gripper_states": np.array(last_gripper_state.width),
            }

            if previous_state_dict is not None:
                for proprio_key in state_dict.keys():
                    proprio_state = state_dict[proprio_key]
                    if np.sum(np.abs(proprio_state)) <= 1e-6:
                        proprio_state = previous_state_dict[proprio_key]
                    state_dict[proprio_key] = np.copy(proprio_state)
            for proprio_key in state_dict.keys():
                data[proprio_key].append(state_dict[proprio_key])

            previous_state_dict = state_dict

            for camera_id in self.camera_ids:
                img_info = cr_interfaces[camera_id].get_img_info()
                print(img_info)
                data[f"camera_{camera_id}"].append(img_info)
                # raise ValueError, "Make sure you debugged the depth image storage."

            end_time = time.time_ns()
            print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

        for camera_id in self.camera_ids:
            cr_interfaces[camera_id].stop()

        self.tmp_folder = folder
        self.tmp_data = data
        return data, folder

    def save(self, keep=True, keyboard_ask=False):
        data = self.tmp_data
        os.makedirs(self.tmp_folder, exist_ok=True)
        with open(f"{self.tmp_folder}/config.json", "w") as f:
            config_dict = {
                "controller_cfg": dict(self.controller_cfg),
                "controller_type": self.controller_type,
                "observation_cfg": self.observation_cfg,
                "intrinsics": self.intrinsics,
                "extrinsics": self.extrinsics,
            }
            json.dump(config_dict, f)

        np.savez(f"{self.tmp_folder}/testing_demo_action", data=np.array(data["action"]))
        np.savez(f"{self.tmp_folder}/testing_demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(f"{self.tmp_folder}/testing_demo_joint_states", data=np.array(data["joint_states"]))
        np.savez(f"{self.tmp_folder}/testing_demo_gripper_states", data=np.array(data["gripper_states"]))

        for camera_id in self.camera_ids:
            np.savez(f"{self.tmp_folder}/testing_demo_camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))

        print("Total length of the trajectory: ", len(data["action"]))

        if keyboard_ask:
            valid_input = False
            while not valid_input:
                try:
                    keep = input(f"Save to {self.tmp_folder} or not? (enter 0 or 1)")
                    keep = bool(int(keep))
                    valid_input = True
                except:
                    pass
            
        
        if not keep:
            import shutil
            shutil.rmtree(f"{self.tmp_folder}")
        self.tmp_folder = None
        self.tmp_data = None
        self.robot_interface.close()
        del self.robot_interface
        return True

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument(
        '--dataset-name',
        type=str,
        default="demonstration_data/example_data"
    )
    return parser.parse_args()
    
def main():

    args = parse_args()
    interface = RedisInteractiveInterface(
        identifier="data_collection", 
        redis_host="172.16.0.1",
        operation_mode_options=["reset", "collect", "stop", "eval"],
        )
    
    observation_cfg = YamlConfig("real_robot_observation_cfg_example.yml").as_easydict()
    observation_cfg.camera_ids = [0]
    data_collection = DeoxysDataCollection(
        observation_cfg,
        folder=args.dataset_name)

    data_collection.collect_data()
    response_data = interface.send_request_and_wait_for_response(request_data={"ask_for_save": True}, target="ui")

    data_collection.save(keep=response_data["data"]["save"])

def local_test():
    args = parse_args()    

    assert(args.dataset_name is not None)
    interface = RedisInteractiveInterface(
        identifier="data_collection", 
        redis_host="172.16.0.1",
        operation_mode_options=["reset", "collect", "stop", "eval"],
        )
    
    observation_cfg = YamlConfig("real_robot_observation_cfg_example.yml").as_easydict()
    observation_cfg.camera_ids = [0]
    data_collection = DeoxysDataCollection(
        observation_cfg,
        folder=args.dataset_name)

    data_collection.collect_data()

    data_collection.save(keep=True, keyboard_ask=True)

   
    
if __name__ == "__main__":
    # main()
    local_test()
