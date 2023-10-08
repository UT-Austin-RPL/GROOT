import os
import torch
import numpy as np
import cv2
import argparse
from functools import partial

from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup

from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import get_autocast_dtype, build_model_for_eval
from dinov2.eval.utils import ModelWithIntermediateLayers

from sklearn.decomposition import PCA

from vos_3d_algo import GROOT_ROOT_PATH
from vos_3d_algo.misc_utils import VideoWriter

class DinoV2ImageProcessor(object):
    def __init__(self, args=None):
        if args is None:
            self.args = EasyDict()
            self.args.output_dir = ''
            self.args.opts = []
            self.args.pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
            self.args.config_file = os.path.join(GROOT_ROOT_PATH, "third_party/dinov2/dinov2/configs/eval/vitb14_pretrain.yaml")
        else:
            self.args = args
        # print("*****")
        print(self.args)
        self.model, self.autocast_dtype = self.setup_and_build_model()
        self.n_last_blocks_list = [1, 4]
        self.n_last_blocks = max(self.n_last_blocks_list)
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=self.autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(self.model, self.n_last_blocks, self.autocast_ctx)

    @staticmethod
    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x

    def setup_and_build_model(self):
        cudnn.benchmark = True
        config = setup(self.args)
        model = build_model_for_eval(config, self.args.pretrained_weights)
        autocast_dtype = get_autocast_dtype(config)
        return model, autocast_dtype

    def process_image(self, img):
        # img = cv2.imread(image_path)
        sizes = [448, 224]
        features = []
        max_size = max(sizes) // 14

        for size in sizes:
            img = cv2.resize(img, (size, size))
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.
            img_tensor = self.color_normalize(img_tensor)
            feature = self.feature_model(img_tensor)[-1][0]
            new_feat = torch.nn.functional.interpolate(rearrange(feature, 'b (h w) c -> b c h w', h=int(np.sqrt(feature.shape[1]))), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
            new_feat = rearrange(new_feat, 'b c h w -> b h w c')
            features.append(new_feat.squeeze(0))

        features = torch.mean(torch.stack(features), dim=0)
        return features
        # return self.pca_transform(features, max_size)

    @staticmethod
    def pca_transform(features, max_size):
        pca = PCA(n_components=3)
        pca_tensor = pca.fit_transform(features.detach().cpu().numpy().reshape(-1, 768))
        pca_tensor = (pca_tensor - pca_tensor.min()) / (pca_tensor.max() - pca_tensor.min())    
        pca_tensor = (pca_tensor * 255).astype(np.uint8).reshape(max_size, max_size, 3)
        return pca_tensor

    def save_image(self, pca_tensor, out_path="dinov2_pca.png"):
        cv2.imwrite(out_path, pca_tensor)

def compute_affinity(feat_1_tuple, feat_2_tuple, temperature=1):
    feat_1, h, w = feat_1_tuple
    feat_2, h2, w2 = feat_2_tuple
    feat_1 = rearrange(feat_1, 'h w c -> (h w) c')
    feat_2 = rearrange(feat_2, 'h w c -> (h w) c')
    sim_matrix = torch.einsum("lc,sc->ls", feat_1, feat_2) / temperature
    aff = sim_matrix
    # aff = F.softmax(aff, dim=0)
    aff = aff.cpu().view(h, w, h2, w2)
    # compute softmax over the first two axes
    return aff

def rescale_feature_map(img_tensor, target_h, target_w, convert_to_numpy=True):
    img_tensor = torch.nn.functional.interpolate(img_tensor, (target_h, target_w))
    if convert_to_numpy:
        return img_tensor.cpu().numpy()
    else:
        return img_tensor
    
def generate_images_from_affinity(source_image, 
                                 target_image, 
                                 source_feature, 
                                 target_feature, 
                                 img_path="./dinov2_features",
                                 h=32,
                                 w=32,
                                 patch_size=14):
    """_summary_

    Args:
        source_image (_type_): _description_
        target_image (_type_): _description_
        source_feature_list (_type_): _description_
        target_feature_list (_type_): _description_
        video_path (str, optional): _description_. Defaults to "./dinov2_tmp_video.mp4".
        h (int, optional): _description_. Defaults to 32.
        w (int, optional): _description_. Defaults to 32.
        patch_size (int, optional): _description_. Defaults to 14.
    """    
    aff = compute_affinity((source_feature, h, w), (target_feature, h, w)) 
    aff = rescale_feature_map(aff, target_image.shape[0], target_image.shape[1])
    img_size = source_image.shape[0]

    count = 0
    for i in range(0, source_image.shape[1] // patch_size):
        for j in range(0, source_image.shape[0] // patch_size):
            source_image_mark = cv2.rectangle(source_image.copy(), 
                                              (i * patch_size, j * patch_size), 
                                              ((i+1) * patch_size, (j+1) * patch_size), 
                                              (0, 0, 1.0), 3)
            
            select_aff = aff[int((j * patch_size) * h / source_image.shape[0]):int(((j+1) * patch_size) * h / source_image.shape[0]),
                             int((i * patch_size) * w / source_image.shape[1]):int(((i+1) * patch_size) * w / source_image.shape[1])]

            select_aff = select_aff.copy().mean(axis=0).mean(axis=0)
            max_y, max_x = np.unravel_index(select_aff.argmax(), select_aff.shape)

            normalized_heatmap = 255 - cv2.normalize(select_aff, None, alpha=0, beta=255, 
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            heatmap_3ch = cv2.merge([normalized_heatmap]*3)
            colormap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

            alpha = 0.5
            overlaid_image = cv2.addWeighted(target_image.copy().astype(np.uint8), 1-alpha, colormap, alpha, 0)
            # overlaid_image = cv2.circle(overlaid_image, (max_x, max_y), 9, (128, 0, 128), -1)

            cv2.imwrite(os.path.join(img_path, f"img_{count}.png"), overlaid_image[..., ::-1])
            count += 1
            # final_image = np.concatenate([cv2.resize(source_image_mark, (img_size, img_size)), overlaid_image], axis=1)
    print("images saved to ", img_path)
    
def generate_video_from_affinity(source_image, 
                                 target_image, 
                                 source_feature, 
                                 target_feature, 
                                 video_path="./",
                                 video_name="./dinov2_tmp_video.mp4", 
                                 h=32,
                                 w=32,
                                 patch_size=14):
    """_summary_

    Args:
        source_image (_type_): _description_
        target_image (_type_): _description_
        source_feature_list (_type_): _description_
        target_feature_list (_type_): _description_
        video_path (str, optional): _description_. Defaults to "./dinov2_tmp_video.mp4".
        h (int, optional): _description_. Defaults to 32.
        w (int, optional): _description_. Defaults to 32.
        patch_size (int, optional): _description_. Defaults to 14.
    """    
    aff = compute_affinity((source_feature, h, w), (target_feature, h, w)) 
    aff = rescale_feature_map(aff, target_image.shape[0], target_image.shape[1])
    img_size = source_image.shape[0]

    video_writer = VideoWriter(video_path, video_name=video_name, save_video=True, fps=5)
    
    for i in range(0, source_image.shape[1] // patch_size):
        for j in range(0, source_image.shape[0] // patch_size):
            source_image_mark = cv2.rectangle(source_image.copy(), 
                                              (i * patch_size, j * patch_size), 
                                              ((i+1) * patch_size, (j+1) * patch_size), 
                                              (0, 0, 1.0), 3)
            
            select_aff = aff[int((j * patch_size) * h / source_image.shape[0]):int(((j+1) * patch_size) * h / source_image.shape[0]),
                             int((i * patch_size) * w / source_image.shape[1]):int(((i+1) * patch_size) * w / source_image.shape[1])]

            select_aff = select_aff.copy().mean(axis=0).mean(axis=0)
            max_y, max_x = np.unravel_index(select_aff.argmax(), select_aff.shape)

            normalized_heatmap = 255 - cv2.normalize(select_aff, None, alpha=0, beta=255, 
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            heatmap_3ch = cv2.merge([normalized_heatmap]*3)
            colormap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

            alpha = 0.5
            overlaid_image = cv2.addWeighted(target_image.copy().astype(np.uint8), 1-alpha, colormap, alpha, 0)
            overlaid_image = cv2.circle(overlaid_image, (max_x, max_y), 9, (128, 0, 128), -1)

            final_image = np.concatenate([cv2.resize(source_image_mark, (img_size, img_size)), overlaid_image], axis=1)
            video_writer.append_image(final_image)

    saved_video_file = video_writer.save(flip=True, bgr=False) 
    return saved_video_file


if __name__ == "__main__":
    setup_args_parser = get_setup_args_parser(add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description="description",
        parents=parents,
        add_help=False,
    )
    args = parser.parse_args()
    dinov2 = DinoV2ImageProcessor()

    feature_list = []
    images = []
    for image_path in ["real_1.png", "real_2.png"]:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
        print(img.shape)
        images.append(img)
        features = dinov2.process_image(img)
        feature_list.append(features)

    h = 32
    w = 32
    aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w))

    patch_size = 14
    frame1_img = images[0]
    frame2_img = images[1]
    print(frame2_img.max())
    aff = rescale_feature_map(aff, frame2_img.shape[0], frame2_img.shape[1])

    img_size = frame1_img.shape[0]

    video_writer = VideoWriter("./", save_video=True, fps=5)
    print(frame1_img.shape)
    for i in range(0, frame1_img.shape[1] // patch_size):
        for j in range(0, frame1_img.shape[0] // patch_size):
            frame1_img_mark = cv2.rectangle(frame1_img.copy(), (i * patch_size, j * patch_size), ((i+1) * patch_size, (j+1) * patch_size), (0, 0, 1.0), 3)
            select_aff = aff[int((j * patch_size) * h / frame1_img.shape[0]):int(((j+1) * patch_size) * h / frame1_img.shape[0]), int((i * patch_size) * w / frame1_img.shape[1]):int(((i+1) * patch_size) * w / frame1_img.shape[1])]

            select_aff = select_aff.copy().mean(axis=0).mean(axis=0)

            max_y, max_x = np.unravel_index(select_aff.argmax(), select_aff.shape)
            # Normalize heatmap values to be in range of 0 to 255
            normalized_heatmap = 255 - cv2.normalize(select_aff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            # Convert heatmap to 3-channel image
            heatmap_3ch = cv2.merge([normalized_heatmap]*3)

            # Apply color map to normalized heatmap
            colormap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

            # Blend color-mapped heatmap with RGB image
            alpha = 0.5  # adjust alpha value to change the heatmap transparency
            overlaid_image = cv2.addWeighted((frame2_img.copy()).astype(np.uint8), 1-alpha, colormap, alpha, 0)

            overlaid_image = cv2.circle(overlaid_image, (max_x, max_y), 9, (128, 0, 128), -1)


            final_image = np.concatenate([cv2.resize(frame1_img_mark, (img_size, img_size)), overlaid_image], axis=1)
            video_writer.append_image(final_image)

    video_writer.save(flip=True, bgr=True) 
