"""
This is a script to verify the camera observations of the real robot. We assume that you are using deoxys_vision for capturing images. If you are using a different vision pipeline, please modify the code accordingly. 
"""

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from deoxys_vision.utils.calibration_utils import load_default_extrinsics, load_default_intrinsics
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info

from real_robot_scripts.groot_img_utils import ImageProcessor
from real_robot_scripts.real_robot_utils import RealRobotObsProcessor

def main():
    # Make sure that you've launched camera nodes somewhere else
    observation_cfg = YamlConfig("real_robot_scripts/real_robot_observation_cfg_example.yml").as_easydict()

    observation_cfg.cameras = []
    for camera_ref in observation_cfg.camera_refs:
        assert_camera_ref_convention(camera_ref)
        camera_info = get_camera_info(camera_ref)

        observation_cfg.cameras.append(camera_info)

    # cr_interfaces = {}
    # for camera_info in observation_cfg.cameras:
    #     cr_interface = CameraRedisSubInterface(camera_info=camera_info)
    #     cr_interface.start()
    #     cr_interfaces[camera_info.camera_name] = cr_interface

    # # type_fn = lambda x: observation_cfg.camera_types[f"camera_{x}"]

    # color_images = []
    # depth_images = []
    # for camera_name in cr_interfaces.keys():
    #     camera_type = observation_cfg.cameras[camera_name].camera_type
    #     camera_id = observation_cfg.cameras[camera_name].camera_id
    #     extrinsics = load_default_extrinsics(camera_id, camera_type, calibration_method="tsai", fmt="dict")
    #     intrinsics = load_default_intrinsics(camera_id, camera_type, fmt="dict")

    #     imgs = cr_interfaces[camera_id].get_img()
    #     img_info = cr_interfaces[camera_id].get_img_info()
    #     color_images.append(imgs['color'])
    #     depth_images.append(imgs['depth'])

        
    obs_processor = RealRobotObsProcessor(observation_cfg,
                                          processor_name="ImageProcessor")

    color_imgs, depth_imgs = obs_processor.get_original_imgs()
    print(color_imgs[0].shape)

    # Get the camera info


    # visualize point clouds using plotly




if __name__ == "__main__":
    main()
