"""Simple wrappers for using Open3D functionalities."""
import cv2
import open3d as o3d
import numpy as np

try:
    from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    from robomimic.utils.obs_utils import Modality

    import robosuite.utils.transform_utils as T
    import robosuite.macros as macros
    from kaede_utils.robosuite_utils.xml_utils import postprocess_model_xml, get_camera_info_from_xml
    from groot_imitation.groot_algo.env_wrapper import rotate_camera
except:
    pass

def convert_convention(image, real_robot=True):
    if not real_robot:
        if macros.IMAGE_CONVENTION == "opencv":
            return np.ascontiguousarray(image[::1])
        elif macros.IMAGE_CONVENTION == "opengl":
            return np.ascontiguousarray(image[::-1])
    else:
        # return np.ascontiguousarray(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return np.ascontiguousarray(image)

class O3DPointCloud():
    def __init__(self, 
                #  robosuite_convention="opengl",
                 max_points=512):
        self.pcd = o3d.geometry.PointCloud()

        self.max_points = max_points
        

    def create_from_rgbd(self, color, depth, intrinsic_matrix, convert_rgb_to_intensity=False):
        """Create a point cloud from RGB-D images.

        Args:
            color (np.ndarray): RGB image.
            depth (np.ndarray): Depth image.
            intrinsic_matrix (np.ndarray): Intrinsic matrix.
            convert_rgb_to_intensity (bool, optional): Whether to convert RGB to intensity. Defaults to False.
        """
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            convert_rgb_to_intensity=convert_rgb_to_intensity)
        
        width, height = color.shape[:2]
        pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix= intrinsic_matrix)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinholecameraIntrinsic)

    def create_from_depth(self, depth, intrinsic_matrix, depth_trunc=5):
        width, height = depth.shape[:2]
        pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix= intrinsic_matrix)

        self.pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), pinholecameraIntrinsic, depth_trunc=depth_trunc)

    def create_from_robosuite(self, color_image, depth_image, env_sim, camera_name="agentview"):
        """Create a point cloud directly from robosuite observation."""
        color = convert_convention(color_image)
        depth = get_real_depth_map(env_sim, convert_convention(depth_image)) * 1000
        intrinsic_matrix = get_camera_intrinsic_matrix(env_sim, camera_name, color.shape[0], color.shape[1])
        extrinsic_matrix = get_camera_extrinsic_matrix(env_sim, camera_name)
        self.create_from_rgbd(color, depth, intrinsic_matrix)
        self.transform(extrinsic_matrix)
        return {"intrinsic_matrix": intrinsic_matrix, 
                "extrinsic_matrix": extrinsic_matrix}
    
    def create_from_points(self, points):
        # points: (num_points, 3)
        self.pcd.points = o3d.utility.Vector3dVector(points)
    
    def project_to_new_camera_frame(self, env_sim, camera_name="agentview"):
        model_xml = env_sim.model.get_xml()
        canonical_camera_pos, canonical_camera_quat = get_camera_info_from_xml(model_xml, camera_name)
        camera_pos, camera_quat = rotate_camera(canonical_camera_pos, canonical_camera_quat, angle=-30, axis=[0, 0, 1], point=[0, 0, 0])
        camera_rot = T.quat2mat(camera_quat)


    def preprocess(self, use_rgb=True):
        num_points = self.get_num_points()

        if num_points < self.max_points:
            num_pad_points = self.max_points - num_points

            if num_pad_points > 0:
                # Randomly select points from the original point cloud for padding
                pad_indices = np.random.randint(0, num_points, size=(num_pad_points,))
                pad_points = self.get_points()[pad_indices]
                if use_rgb:
                    pad_colors = self.get_colors()[pad_indices]
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(pad_points)
                if use_rgb:
                    new_pcd.colors = o3d.utility.Vector3dVector(pad_colors)
                self.pcd += new_pcd
        else:
            self.pcd = self.pcd.random_down_sample(self.max_points / num_points)
            # In case downsampling results in fewer points
            if self.get_num_points() < self.max_points:
                self.preprocess(use_rgb=use_rgb)

    def merge(self, other):
        """Merge two point clouds.

        Args:
            other (O3DPointCloud): Another point cloud.
        """
        self.pcd += other.pcd

    def transform(self, extrinsic_matrix):
        """Transform the point cloud.

        Args:
            extrinsic_matrix (np.ndarray): Extrinsic matrix.
        """
        return self.pcd.transform(extrinsic_matrix)
    
    def save(self, file_name, verbose=False):
        """Save the point cloud.

        Args:
            file_name (str): File name.
        """
        o3d.io.write_point_cloud(file_name, self.pcd)
        if verbose:
            print(f"Saved point cloud to {file_name}.")

    def get_points(self):
        """Get the points.

        Returns:
            np.ndarray: (num_points, 3), where each point is (x, y, z).
        """
        return np.asarray(self.pcd.points)
    
    def get_num_points(self):
        """Get the number of points.

        Returns:
            int: Number of points.
        """
        return len(self.get_points())
    
    def get_colors(self):
        """Get the colors.

        Returns:
            np.ndarray: (num_points, 3), where each color is (r, g, b).
        """
        return np.asarray(self.pcd.colors)


def create_partial_view_from_camera(o3d_point_cloud, new_camera_pos, radius=3000):
    _, pt_map = o3d_point_cloud.pcd.hidden_point_removal(new_camera_pos, radius=radius)
    new_pcd = o3d_point_cloud.pcd.select_by_index(pt_map)
    new_o3d_pcd = O3DPointCloud(o3d_point_cloud.max_points)
    new_o3d_pcd.pcd = new_pcd
    return new_o3d_pcd