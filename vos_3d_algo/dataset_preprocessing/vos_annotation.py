"""Get all the masks for the demonstration dataset"""
import h5py
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from kaede_utils.visualization_utils.video_utils import KaedeVideoWriter

import init_path
from vos_3d_algo.xmem_tracker import XMemTracker
from vos_3d_algo.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter
from vos_3d_algo.o3d_modules import convert_convention

# def parse_args():
#     args = argparse.ArgumentParser(description='Get all the masks for the demonstration dataset')
#     args.add_argument('--dataset', type=str, default='data/demonstration_dataset.hdf5', help='path to the demonstration dataset')
#     # visualize video for debugging
#     args.add_argument('--save-video', action='store_true', help='visualize the video')
#     # see if it's real data
#     args.add_argument('--real', action='store_true', help='real data')
#     args.add_argument('--multi-instance', action='store_true', help='multi-instance case')
#     args.add_argument('--verbose', action='store_true', help='verbose')
#     return args.parse_args()

def dataset_vos_annotation(cfg,
                           dataset_name,
                           mask_dataset_name,
                           xmem_tracker,
                           annotation_folder,
                           save_video=True,
                           is_real_robot=True,
                           verbose=False,
                           ):
    """This is the case where we only focus on manipulation one specific-instance."""
    
    if save_video:
        video_writer = VideoWriter(annotation_folder, "annotation_video.mp4", fps=40.0)

    # first_frame = cv2.imread(os.path.join(annotation_folder, "frame.jpg"))
    # first_frame = first_frame[:, :, ::-1]
    # first_frame_annotation = np.array(Image.open((os.path.join(annotation_folder, "frame_annotation.png"))))
    first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)
    with h5py.File(dataset_name, 'r') as dataset, h5py.File(mask_dataset_name, 'w') as new_dataset:

        # TODO: Speciify if it's a multi-instance case
        count = 0
        new_dataset.create_group("data")
        for demo in tqdm(dataset["data"].keys()):
            xmem_tracker.clear_memory()
            if verbose:
                print("processing demo: ", demo)
            images = dataset[f"data/{demo}/obs/agentview_rgb"][()]
            image_list = [first_frame]
            for image in images:
                image_list.append(convert_convention(image, real_robot=is_real_robot))
            image_list = [cv2.resize(image, (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA) for image in image_list]
            masks = xmem_tracker.track_video(image_list, first_frame_annotation)

            if verbose:
                print(len(image_list), len(masks))

            new_dataset.create_group(f"data/{demo}/obs")
            new_dataset[f"data/{demo}/obs"].create_dataset("agentview_masks", data=np.stack(masks[1:], axis=0))
            assert(len(masks[1:]) == len(images))
            if save_video:
                overlay_images = []
                for rgb_img, mask in zip(image_list, masks):
                    colored_mask = Image.fromarray(mask)
                    colored_mask.putpalette(xmem_tracker.palette)
                    colored_mask = np.array(colored_mask.convert("RGB"))
                    overlay_img = cv2.addWeighted(rgb_img, 0.7, colored_mask, 0.3, 0)

                    overlay_images.append(overlay_img)
                    video_writer.append_image(overlay_img)

    if save_video:
        video_writer.save(flip=True, bgr=False)
