"""Miscellaneous utility functions."""

import os
import cv2
import h5py
import numpy as np

from PIL import Image
from third_party.XMem.util.palette import davis_palette

from kaede_utils.visualization_utils.video_utils import KaedeVideoWriter

def get_annotation_path(dataset_name, parent_folder="annotations"):
    dataset_folder_name = dataset_name.split("/")[-1].replace(".hdf5", "")
    annotations_folder = os.path.join(parent_folder, dataset_folder_name)
    return annotations_folder


def get_first_frame_annotation(annotations_folder):
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

        video_writer = KaedeVideoWriter(video_path=video_path, fps=30, save_video=True)
        for overlay_img in overlay_images:
            video_writer.append_image(overlay_img)
        video_writer.save(video_name, flip=flip)
    return video_path

def get_first_frame_annotation_from_dataset(dataset_name):
    with h5py.File(dataset_name, "r") as dataset:
        first_frame = dataset["annotation"]["first_frame"][()]
        first_frame_annotation = dataset["annotation"]["first_frame_annotation"][()]
    return first_frame, first_frame_annotation

def get_palette():
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
    return davis_palette

def overlay_xmem_mask_on_image(rgb_img, mask):
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    overlay_img = cv2.addWeighted(rgb_img, 0.7, colored_mask, 0.3, 0)

    return overlay_img

def mask_to_rgb(mask):
    """Make sure this mask is directly taken from `xmem_tracker.track`"""
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    return colored_mask
