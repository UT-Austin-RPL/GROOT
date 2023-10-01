"""Miscellaneous utility functions."""

import os
import cv2
import imageio
import h5py
import numpy as np
import plotly.express as px

from PIL import Image
from third_party.XMem.util.palette import davis_palette

def get_annotation_path(dataset_name, parent_folder="annotations"):
    dataset_folder_name = dataset_name.split("/")[-1].replace(".hdf5", "")
    annotations_folder = os.path.join(parent_folder, dataset_folder_name)
    return annotations_folder


def get_first_frame_annotation(annotations_folder):
    """A helper function to get the first frame and its annotation from the specified annotations folder.

    Args:
        annotations_folder (str): path to where the annotation is stored

    Returns:
        image, annotated_mask
    """
    first_frame = cv2.imread(os.path.join(annotations_folder, "frame.jpg"))[:, :, ::-1]
    first_frame_annotation = np.array(Image.open((os.path.join(annotations_folder, "frame_annotation.png"))))
    # Resize first_frame to first_frame_annotation if shape does not match
    if first_frame.shape[0] != first_frame_annotation.shape[0] or first_frame.shape[1] != first_frame_annotation.shape[1]:
        first_frame = cv2.resize(first_frame, (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA)
    return first_frame, first_frame_annotation

def get_overlay_video_from_dataset(dataset_name, demo_idx=None, palette=davis_palette, video_name="overlay_video.mp4", flip=True):
    annotations_folder = get_annotation_path(dataset_name)
    video_path = annotations_folder

    with h5py.File(dataset_name, "r") as original_dataset, h5py.File(os.path.join(annotations_folder, "masks.hdf5"), "r") as mask_dataset:
        demo_keys = [f"demo_{demo_idx}" if demo_idx is not None else demo for demo in original_dataset["data"].keys()]
        overlay_images = []
        for demo in demo_keys:
            images = original_dataset[f"data/{demo}/obs/agentview_rgb"][()]
            masks = mask_dataset[f"data/{demo}/obs/agentview_masks"][()]
            for (image, mask) in zip(images, masks):
                colored_mask = Image.fromarray(mask)
                colored_mask.putpalette(palette)
                colored_mask = np.array(colored_mask.convert("RGB"))
                # resize image to colored_mask
                image = cv2.resize(image, (colored_mask.shape[1], colored_mask.shape[0]), interpolation=cv2.INTER_AREA)
                overlay_img = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                overlay_images.append(overlay_img)

        video_writer = VideoWriter(video_path=video_path, fps=30, save_video=True)
        for overlay_img in overlay_images:
            video_writer.append_image(overlay_img)
        video_writer.save(video_name, flip=flip)
    return video_path

def get_first_frame_annotation_from_dataset(dataset_name):
    with h5py.File(dataset_name, "r") as dataset:
        first_frame = dataset["annotation"]["first_frame"][()]
        first_frame_annotation = dataset["annotation"]["first_frame_annotation"][()]
    return first_frame, first_frame_annotation

def get_palette(palette="davis"):
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
    youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'
    if palette == "davis":
        return davis_palette
    elif palette == "youtube":
        return youtube_palette

def overlay_xmem_mask_on_image(rgb_img, mask, use_white_bg=False):
    """

    Args:
        rgb_img (np.ndarray):rgb images
        mask (np.ndarray)): binary mask
        use_white_bg (bool, optional): Use white backgrounds to visualize overlap. Note that we assume mask ids 0 as the backgrounds. Otherwise the visualization might be screws up. . Defaults to False.

    Returns:
        np.ndarray: overlay image of rgb_img and mask
    """
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    if use_white_bg:
        colored_mask[mask == 0] = [255, 255, 255]
    overlay_img = cv2.addWeighted(rgb_img, 0.7, colored_mask, 0.3, 0)

    return overlay_img

def mask_to_rgb(mask):
    """Make sure this mask is directly taken from `xmem_tracker.track`"""
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    return colored_mask

def depth_to_rgb(depth_image, colormap="jet"):
    # Normalize depth values between 0 and 1
    normalized_depth = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Apply a colormap to the normalized depth image
    if colormap == "jet":
        colormap = cv2.COLORMAP_JET
    elif colormap == "magma":
        colormap = cv2.COLORMAP_MAGMA
    elif colormap == "viridis":
        colormap = cv2.COLORMAP_VIRIDIS
    else:
        raise ValueError(f"Unknown colormap: {colormap}. Please choose from 'jet', 'magma', or 'viridis'")

    depth_colormap = cv2.applyColorMap(np.uint8(normalized_depth * 255), colormap)
    return depth_colormap

def set_pillow_image_alpha(img, alpha):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    assert(img.mode == "RGBA"), f"Image mode must be 'RGBA', but got {img.mode} instead"
    alpha = int(alpha * 255)
    img.putalpha(alpha)
    return img

def add_palette_on_mask(mask_img, palette="davis", alpha=1.0):
    if isinstance(mask_img, np.ndarray):
        assert(len(mask_img.shape) == 2 or mask_img.shape[2] == 1), f"The mask image needs to be a single channel image, but got shape {mask_img.shape} instead"
        mask_img = Image.fromarray(mask_img)
    
    assert(len(np.array(mask_img).shape) == 2 or np.array(mask_img).shape[2] == 1), f"The mask image needs to be a single channel image, but got shape {mask_img.shape} instead"
    # copy mask_img
    new_mask_img = mask_img.copy()
    if palette == "davis":
        new_mask_img.putpalette(get_palette(palette="davis"))
    elif palette == "youtube":
        new_mask_img.putpalette(get_palette(palette="youtube"))
    else:
        raise ValueError(f"Unknown palette: {palette}. Please choose from 'davis' or 'youtube'")
    new_mask_img = new_mask_img.convert("RGBA")
    if alpha < 1.0:
        new_mask_img = set_pillow_image_alpha(new_mask_img, alpha)
    return new_mask_img

def resize_image_to_same_shape(source_img, reference_img=None, reference_size=None):
    # if source_img is larger than reference_img
    if reference_img is None and reference_size is None:
        raise ValueError("Either reference_img or reference_size must be specified.")
    if reference_img is not None:
        reference_size = (reference_img.shape[0], reference_img.shape[1])
    if source_img.shape[0] >  reference_size[0] or source_img.shape[1] > reference_size[1]:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_NEAREST)
    else:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_AREA)
    return result_img


def plotly_draw_seg_image(image, mask):
    fig = px.imshow(image)

    fig.data[0].customdata = mask
    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}<br>Mask ID: %{customdata}'


    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=300,   # you can adjust this as needed
        height=300,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()


def edit_h5py_datasets(base_dataset_name, additional_dataset_name, mode="merge  "):
    # load base dataset in an edit mode
    base_dataset = h5py.File(base_dataset_name, "r+")
    additional_dataset = h5py.File(additional_dataset_name, "r")

    # check if the additional dataset is compatible with the base dataset
    for demo in base_dataset["data"].keys():
        assert(demo in additional_dataset["data"].keys()), f"Demo {demo} does not exist in the additional dataset."
    try:
        # merge mode
        if mode == "merge":
            for demo in base_dataset["data"].keys():
                print(demo)
                for key in additional_dataset[f"data/{demo}/obs"].keys():
                    # if key in base_dataset[f"data/{demo}/obs"].keys():
                    #     print("Warning")
                    #     continue
                    assert(key not in base_dataset[f"data/{demo}/obs"].keys()), f"Key {key} already exists in the base dataset."
                    additional_dataset.copy(additional_dataset[f"data/{demo}/obs/{key}"], base_dataset[f"data/{demo}/obs"], key)

        # separate mode
        if mode == "separate":
            for demo in base_dataset["data"].keys():
                for key in additional_dataset[f"data/{demo}/obs"].keys():
                    assert(key in base_dataset[f"data/{demo}/obs"].keys()), f"Key {key} does not exist in the base dataset."
                    del base_dataset[f"data/{demo}/obs/{key}"]
    except Exception as e:
        print(e)
        base_dataset.close()
        additional_dataset.close()
        raise e

    base_dataset.close()
    additional_dataset.close()


class VideoWriter():
    def __init__(self, video_path, save_video=False, video_name=None, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.single_video = single_video
        self.last_images = {}
        if video_name is None:
            self.video_name = "video.mp4"
        else:
            self.video_name = video_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self.video_name)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_image(self, image, idx=0):
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            self.image_buffer[idx].append(image[::-1])

    def append_vector_obs(self, images):
        if self.save_video:
            for i in range(len(images)):
                self.append_image(images[i], i)

    def save(self, video_name=None, flip=True, bgr=True):
        if video_name is None:
            video_name = self.video_name
        img_convention = 1
        color_convention = 1
        if flip:
            img_convention = -1
        if bgr:
            color_convention = -1
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, video_name)
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                    video_writer.close()
            print(f"Saved videos to {video_name}.")
        return os.path.join(self.video_path, video_name)



            