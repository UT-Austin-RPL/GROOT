"""
This is a script to verify the camera observations of the real robot. We assume that you are using deoxys_vision for capturing images. If you are using a different vision pipeline, please modify the code accordingly. 
"""

import plotly.graph_objs as go

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

import init_path
from groot_imitation.groot_algo.dataset_preprocessing.pcd_generation import scene_pcd_fn

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
    obs_processor.load_intrinsic_matrix(resize=False)
    obs_processor.load_extrinsic_matrix()
    extrinsic_matrix = obs_processor.get_extrinsic_matrix("agentview")
    intrinsic_matrix = obs_processor.get_intrinsic_matrix("agentview")

    color_imgs, depth_imgs = obs_processor.get_original_imgs()
    print(color_imgs[0].shape)

    pcd_points, pcd_colors = scene_pcd_fn(
        observation_cfg,
        rgb_img_input=color_imgs[0],
        depth_img_input=depth_imgs[0],
        extrinsic_matrix=extrinsic_matrix,
        intrinsic_matrix=intrinsic_matrix,
    )    

    # visualize point clouds using plotly
    color_str = ['rgb('+str(r)+','+str(g)+','+str(b)+')' for r,g,b in pcd_colors]

    # Extract x, y, and z columns from the point cloud
    x_vals = pcd_points[:, 0]
    y_vals = pcd_points[:, 1]
    z_vals = pcd_points[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(size=3, color=color_str, opacity=0.8)
    )

    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[rgbd_scatter], layout=layout)

    # Show the figure
    fig.show()

    # Get the camera info






if __name__ == "__main__":
    main()
