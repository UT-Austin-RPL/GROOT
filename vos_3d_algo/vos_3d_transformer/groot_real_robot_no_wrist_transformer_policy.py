import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange

from libero.lifelong.models.modules.data_augmentation import *
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import PolicyMeta
from libero.lifelong.models.policy_head import *

from vos_3d_algo.modules import *
from vos_3d_algo.point_mae_modules import *


###############################################################################
#
# A model handling extra input modalities besides images at time t.
#
###############################################################################

class ExtraModalityTokens(nn.Module):
    def __init__(self,
                 use_joint=False,
                 use_gripper=False,
                 use_ee=False,
                 extra_num_layers=0,
                 extra_hidden_size=64,
                 extra_embedding_size=32):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 1
        ee_dim = 6

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)

        extra_low_level_feature_dim = int(use_joint) * joint_states_dim + \
                int(use_gripper) * gripper_states_dim + \
                int(use_ee) * ee_dim

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = {}

        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0 # we indeed have extra information
            if extra_num_layers > 0:
                layers = [
                    nn.Linear(extra_low_level_feature_dim, extra_hidden_size)
                ]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True)
                    ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [
                    nn.Linear(extra_low_level_feature_dim, extra_embedding_size)
                ]

            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for (proprio_dim, use_modality, modality_name) in [
                (joint_states_dim, self.use_joint, "joint_states"),
                (gripper_states_dim, self.use_gripper, "gripper_states"),
                (ee_dim, self.use_ee, "ee_states")]:

            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)
            
        self.encoders = nn.ModuleList([
            x["encoder"] for x in self.extra_encoders.values()])

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3) 
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for (use_modality, modality_name) in [
                (self.use_joint, "joint_states"),
                (self.use_gripper, "gripper_states"),
                (self.use_ee, "ee_states")]:

            if use_modality:
                tensor_list.append(self.extra_encoders[modality_name]["encoder"](obs_dict[modality_name]))
        
        x = torch.stack(tensor_list, dim=-2)
        return x

###############################################################################
#
# A Transformer Policy
#
###############################################################################

class GROOTRealRobotNoWristTransformerPolicy(nn.Module, metaclass=PolicyMeta):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """
    def __init__(self,
                 cfg, 
                 shape_meta):
        super().__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.shape_meta = shape_meta

        policy_cfg = cfg.policy
        
        ### 1. encode image
        embed_size = policy_cfg.embed_size
        transformer_input_sizes = []

        self.pcd_encoders = {}

        for name in shape_meta["all_shapes"].keys():
            if "xyz" == name or "rgb" == name:
                kwargs = policy_cfg.pcd_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                # Currently let's assume all the input features are merged
                # kwargs.input_shape = kwargs.input_shape[-2] + shape_meta["all_shapes"]["rgb"][-2]
                kwargs.output_size = embed_size
                kwargs.language_dim = policy_cfg.language_encoder.network_kwargs.input_size
                self.pcd_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.pcd_encoder.network)(**kwargs)
                }

        self.encoders = nn.ModuleList([x["encoder"] for x in self.pcd_encoders.values()])
        self.pcd_aug = eval(policy_cfg.pcd_aug.network)(**policy_cfg.pcd_aug.network_kwargs)
        # ### 2. encode language
        # policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        # self.language_encoder = eval(policy_cfg.language_encoder.network)(
        #         **policy_cfg.language_encoder.network_kwargs)

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(
                 use_joint=cfg.data.use_joint,
                 use_gripper=cfg.data.use_gripper,
                 use_ee=cfg.data.use_ee,
                 extra_num_layers=policy_cfg.extra_num_layers,
                 extra_hidden_size=policy_cfg.extra_hidden_size,
                 extra_embedding_size=embed_size)

        ### 4. define temporal transformer
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
                policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer = TransformerDecoder(
                 input_size=embed_size,
                 num_layers=policy_cfg.transformer_num_layers,
                 num_heads=policy_cfg.transformer_num_heads,
                 head_output_size=policy_cfg.transformer_head_output_size,
                 mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
                 dropout=policy_cfg.transformer_dropout,
        )

        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
                **policy_cfg.policy_head.loss_kwargs,
                **policy_cfg.policy_head.network_kwargs)

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        action_token = nn.Parameter(torch.randn(embed_size))
        self.register_parameter("action_token", action_token)

        # color_aug = eval(policy_cfg.color_aug.network)(
        #         **policy_cfg.color_aug.network_kwargs)



    def temporal_encode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1) # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2) # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:,:,0] # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"]) # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        # text_encoded = self.language_encoder(data) # (B, E)
        # text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1) # (B, T, 1, E)
        encoded = [self.action_token.view(1, 1, 1, -1).expand(B, T, -1, -1)] # (B, T, 1, E)
        encoded.append(extra)

        pcd_features = []
        for name in self.pcd_encoders.keys():
            x = data["obs"][name]
            B, T, O, N, D = x.shape
            # x = rearrange(x, 'B T O N D -> B (T O N) D')
            x = self.pcd_aug(x)
            if self.training:
                suffix = f"{self.cfg.policy.pcd_encoder.network_kwargs.group_cfg.num_group}_{self.cfg.policy.pcd_encoder.network_kwargs.group_cfg.group_size}"
                x1, x2 = data["obs"][f"neighborhood_{suffix}"], data["obs"][f"centers_{suffix}"]
                
                # mask out the last robot

                x1 = rearrange(x1, "b t (o g) n d -> b t o g n d", b=B, t=T, o=O)
                x2 = rearrange(x2, "b t (o g) n d -> b t o g n d", b=B, t=T, o=O)

                x1 = x1[:, :, :-1, ...]
                x2 = x2[:, :, :-1, ...]  
     
                x1 = rearrange(x1, "b t o g n d -> (b t o) g n d", b=B, t=T, o=O-1)
                x2 = rearrange(x2, "b t o g n d -> (b t o) g n d", b=B, t=T, o=O-1).squeeze(-2)

                x, _ = self.pcd_encoders[name]["encoder"].forward_mask(x1, x2)
            else:
                # x = x[:, :, :-1, ...]
                x = rearrange(x, 'B T O N D -> (B T O) D N', T=T, O=O, N=N)
                
                x1, x2 = self.pcd_encoders[name]["encoder"].forward_group(x.contiguous())

                # import pdb; pdb.set_trace()
                x1 = rearrange(x1, "(b t o) g n d -> b t o g n d", b=B, t=T, o=O)
                x2 = rearrange(x2, "(b t o) n d -> b t o n d", b=B, t=T, o=O)
                x1 = x1[:, :, :-1, ...]
                x2 = x2[:, :, :-1, ...]
                x1 = rearrange(x1, "b t o g n d -> (b t o) g n d", b=B, t=T, o=O-1)
                x2 = rearrange(x2, "b t o n d -> (b t o) n d", b=B, t=T, o=O-1)

                x, _ = self.pcd_encoders[name]["encoder"].forward_mask(x1, x2)

            x = rearrange(x, '(B T O) D N-> B T (O D) N', B=B, T=T, O=O-1)
            pcd_features.append(x)
        pcd_features = torch.cat(pcd_features, dim=-2)
        encoded.append(pcd_features)

        # # eye in hand feature
        # wrist_feature = data["obs"]["eye_in_hand_depth"]
        # B, T, C, H, W = wrist_feature.shape
        # wrist_feature = rearrange(wrist_feature, 'B T C H W -> (B T) C H W')
        # wrist_feature = self.wrist_depth_encoder(wrist_feature)
        # wrist_feature = rearrange(wrist_feature, '(B T) D -> B T D', B=B, T=T)
        # wrist_feature = wrist_feature.unsqueeze(-2)
        # encoded.append(wrist_feature)

        encoded = torch.cat(encoded, -2) # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        x = self.spatial_encode(data)
        x = self.temporal_encode(x)
        dist = self.policy_head(x)
        return dist
    
    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            try:
                x = torch.cat(self.latent_queue, dim=1) # (B, T, H_all)
            except:
                import pdb; pdb.set_trace()
            x = self.temporal_encode(x)
            dist = self.policy_head(x[:,-1])
        action = dist.sample().detach().cpu()
        # return action.view(action.shape[0], -1).numpy()
        return action.numpy().squeeze()

    def reset(self):
        self.latent_queue = []


    def _get_img_tuple(self, data):
        img_tuple = tuple([
            data["obs"][img_name] for img_name in self.image_encoders.keys()
        ])
        return img_tuple        

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx] for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode: # apply augmentation
            if self.cfg.train.use_augmentation:
                # data["obs"]["eye_in_hand_depth"] = self.img_aug(data["obs"]["eye_in_hand_depth"])
                pass
                # img_tuple = self._get_img_tuple(data)
                # aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                # for img_name in self.wrist_.keys():
                #     data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(data, {
                torch.Tensor: lambda x: x.unsqueeze(dim=1) # add time dimension
            })
            data["task_emb"] = data["task_emb"].squeeze(1)
        return data

    def compute_loss(self, data, reduction='mean'):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        return loss

    def reset(self):
        """
            Clear all "history" of the policy if there exists any.
        """
        pass
