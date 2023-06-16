import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from pointnet.model import PointNetfeat

from einops import rearrange

class PointNetEncoder(nn.Module):
    """Trying with the most simple version."""
    def __init__(self, input_shape, output_size, global_feat=True, **kwargs):
        super().__init__()

        # self.layers = nn.Sequential(
        #     PointNetfeat(global_feat=global_feat).cuda(),
        #     nn.Linear(1024, output_size)
        # )
        self.pointnet_backbone = PointNetfeat(global_feat=global_feat)
        # linear projection from output of PointNet to feature_dim
        self.linear_projection = nn.Linear(1024, output_size)

        self.input_shape = input_shape
        self.output_size = output_size

    def forward(self, x):
        # x: (B, 3, N)
        # feature: (B, feature_dim)
        feature, _, _ = self.pointnet_backbone(x)
        feature = self.linear_projection(feature)
        return feature
    

class SE3Augmentation(nn.Module):
    def __init__(self, 
                 mean=0.0, 
                 std=0.02, 
                 enabled=True, 
                 use_position=True,
                 use_rotation=False,
                 rot_range=(np.pi / 3, np.pi / 3, np.pi / 3)):
        super().__init__()
        self.mean = mean
        self.std = std
        self.enabled = enabled
        self.use_position = use_position
        self.use_rotation = use_rotation

        self.placeholder_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.rot_x, self.rot_y, self.rot_z = rot_range

    def forward(self, point_set):
        # Only apply augmentation if training
        if self.training and self.enabled:
            # theta = torch.rand(1) * 2 * np.pi
            # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
            #                                 [torch.sin(theta), torch.cos(theta)]]).to(point_set.device)
            point_shape = point_set.shape
            start_dims = np.arange(len(point_shape) - 2).tolist()
            point_set = point_set.permute(start_dims + [-1, -2])
            if self.use_rotation:
                rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((np.random.uniform(-self.rot_x, self.rot_x), np.random.uniform(-self.rot_y, self.rot_y), np.random.uniform(-self.rot_z, self.rot_z)))
                rotation_matrix = torch.from_numpy(np.array(rotation_matrix)).float().to(point_set.device)
                point_set[..., :] = point_set[..., :].matmul(rotation_matrix)
            if self.use_position:
                point_set += torch.normal(self.mean, self.std, size=point_set.shape).to(point_set.device)
            point_set = point_set.permute(start_dims + [-1, -2])
        elif False: # self.enabled:
            # see if rotation augmentation needs to be turned on during evaluation
            point_shape = point_set.shape
            start_dims = np.arange(len(point_shape) - 2).tolist()
            point_set = point_set.permute(start_dims + [-1, -2])
            if self.use_rotation:
                rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((np.random.uniform(-self.rot_x, self.rot_x), np.random.uniform(-self.rot_y, self.rot_y), np.random.uniform(-self.rot_z, self.rot_z)))
                rotation_matrix = torch.from_numpy(np.array(rotation_matrix)).float().to(point_set.device)
                point_set[..., :] = point_set[..., :].matmul(rotation_matrix)
            point_set = point_set.permute(start_dims + [-1, -2])
        return point_set
    
class SE3Augmentation2(nn.Module):
    def __init__(self, 
                 mean=0.0, 
                 std=0.02, 
                 enabled=True, 
                 use_position=True,
                 use_rotation=True,
                 location_range=(1., 1., 1.),
                 rot_range=(np.pi / 3, np.pi / 3, np.pi / 3)):
        super().__init__()
        self.mean = mean
        self.std = std
        self.enabled = enabled
        self.use_position = use_position
        self.use_rotation = use_rotation

        self.placeholder_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.rot_x, self.rot_y, self.rot_z = rot_range
        self.location_range = torch.tensor(location_range)

    def forward(self, point_set):
        # Only apply augmentation if training
        if self.training and self.enabled:
            # theta = torch.rand(1) * 2 * np.pi
            # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
            #                                 [torch.sin(theta), torch.cos(theta)]]).to(point_set.device)
            point_shape = point_set.shape
            start_dims = np.arange(len(point_shape) - 2).tolist()
            point_set = point_set.permute(start_dims + [-1, -2])
            if self.use_rotation:
                rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((np.random.uniform(-self.rot_x, self.rot_x), np.random.uniform(-self.rot_y, self.rot_y), np.random.uniform(-self.rot_z, self.rot_z)))
                rotation_matrix = torch.from_numpy(np.array(rotation_matrix)).float().to(point_set.device)
                point_set[..., :] = point_set[..., :].matmul(rotation_matrix)
                translation = torch.rand(3) * self.location_range
                point_set[..., :] += translation.to(point_set.device)
            if self.use_position:
                point_set += torch.normal(self.mean, self.std, size=point_set.shape).to(point_set.device)
            point_set = point_set.permute(start_dims + [-1, -2])
        return point_set
    
    
class SE3Augmentation3(nn.Module):
    def __init__(self, 
                 mean=0.0, 
                 std=0.02, 
                 enabled=True, 
                 use_position=True,
                 use_rotation=True,
                 location_range=(1., 1., 1.),
                 rot_range=(np.pi / 3, np.pi / 3, np.pi / 3)):
        super().__init__()
        self.mean = mean
        self.std = std
        self.enabled = enabled
        self.use_position = use_position
        self.use_rotation = use_rotation

        self.placeholder_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.rot_x, self.rot_y, self.rot_z = rot_range
        self.location_range = torch.tensor(location_range)

        rot_x_range = np.linspace(-self.rot_x, self.rot_x, 10)
        rot_y_range = np.linspace(-self.rot_y, self.rot_y, 10)
        rot_z_range = np.linspace(-self.rot_z, self.rot_z, 10)

        rotation_matrices = []
        for rot_x in rot_x_range:
            for rot_y in rot_y_range:
                for rot_z in rot_z_range:
                    rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((rot_x, rot_y, rot_z))
                    rotation_matrices.append(torch.from_numpy(np.array(rotation_matrix)).float())
        for i in range(10):
            rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((0, 0, 0))
            rotation_matrices.append(torch.from_numpy(np.array(rotation_matrix)).float())

        rotation_matrices = torch.stack(rotation_matrices)
        self.register_buffer('rotation_matrices', rotation_matrices)


    def forward(self, point_set):
        # Only apply augmentation if training
        if self.training and self.enabled:
            # theta = torch.rand(1) * 2 * np.pi
            # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
            #                                 [torch.sin(theta), torch.cos(theta)]]).to(point_set.device)
            point_shape = point_set.shape
            batch_size = point_shape[0]
            # start_dims = np.arange(len(point_shape) - 2).tolist()
            # point_set = point_set.permute(start_dims + [-1, -2])
            B, T, O, N, D = point_shape
            point_set = rearrange(point_set, 'B T O N D -> B (T O D) N')
            if self.use_rotation:
                self.select_indices = torch.randint(0, len(self.rotation_matrices), (batch_size,))
                rotation_matrix = self.rotation_matrices[self.select_indices]
                point_set = torch.bmm(point_set, rotation_matrix)
                translation = torch.rand(batch_size, 3) * self.location_range
                point_set[..., :] += translation.to(point_set.device).unsqueeze(1)
            if self.use_position:
                point_set += torch.normal(self.mean, self.std, size=point_set.shape).to(point_set.device)
            # point_set = point_set.permute(start_dims + [-1, -2])
            point_set = rearrange(point_set, 'B (T O D) N -> B T O N D', T=T, O=O, D=D)
        return point_set

class SE3ConstrastiveAugmentation(nn.Module):
    def __init__(self, 
                 mean=0.0, 
                 std=0.02, 
                 enabled=True, 
                 use_position=True,
                 use_rotation=False,
                 rot_range=(np.pi / 3, np.pi / 3, np.pi / 3),
                 num_constrast=4):
        super().__init__()
        self.mean = mean
        self.std = std
        self.enabled = enabled
        self.use_position = use_position
        self.use_rotation = use_rotation

        self.num_constrast = num_constrast

        self.placeholder_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.rot_x, self.rot_y, self.rot_z = rot_range

    def forward(self, point_set):
        # Only apply augmentation if training
        if self.training and self.enabled:
            point_shape = point_set.shape
            start_dims = np.arange(len(point_shape) - 2).tolist()
            point_set = point_set.permute(start_dims + [-1, -2])
            if self.use_rotation:
                rotation_matrix = self.placeholder_mesh.get_rotation_matrix_from_xyz((np.random.uniform(-self.rot_x, self.rot_x), np.random.uniform(-self.rot_y, self.rot_y), np.random.uniform(-self.rot_z, self.rot_z)))
                rotation_matrix = torch.from_numpy(np.array(rotation_matrix)).float().to(point_set.device)
                point_set[..., :] = point_set[..., :].matmul(rotation_matrix)
            if self.use_position:
                point_set += torch.normal(self.mean, self.std, size=point_set.shape).to(point_set.device)
            point_set = point_set.permute(start_dims + [-1, -2])
        return point_set
    