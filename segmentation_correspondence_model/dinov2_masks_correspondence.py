# Read the first frame annotation, and compute the dino attention over the first frame
import os
import argparse
import h5py
import cv2
import torch

import numpy as np
from collections import Counter
from einops import rearrange
from pathlib import Path
from PIL import Image
import shutil

import init_path
from vos_3d_algo.misc_utils import get_annotation_path, get_first_frame_annotation, get_first_frame_annotation_from_dataset

from auto_annotation.annotation_utils import get_first_frame_folder, get_first_frame_segmentation_folder

from vos_3d_algo.dino_features import DinoV2ImageProcessor, compute_affinity, rescale_feature_map


def parse_args():
    parser = argparse.ArgumentParser(description='Get all the masks for the demonstration dataset')
    parser.add_argument('--reference-folder', type=str, default='segmentation_correspondence_model/reference_images', help='path to the demonstration dataset')
    parser.add_argument('--target-folder', type=str, default='segmentation_correspondence_model/target_images', help='path to the demonstration dataset')
    # output folder
    parser.add_argument('--output-folder', type=str, default='segmentation_correspondence_model/correspondence_results', help='path to the demonstration dataset')
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    # test
    return parser.parse_args()


def compute_iou(reference_mask, segmentation_masks):
    # Convert to binary arrays
    reference_mask = reference_mask.astype(bool)
    segmentation_masks = segmentation_masks.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(reference_mask, segmentation_masks)
    union = np.logical_or(reference_mask, segmentation_masks)

    # Compute IoU
    iou = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
    return iou

def get_palette():
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'    
    return davis_palette

def find_most_repeated_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else -1

def main():
    args = parse_args()

    # Load the target images and its reference images
    task_names = []
    for task_name in Path(args.target_folder).glob("*"):
        task_names.append(str(task_name).split("/")[-1])

    target_images = {}
    reference_images = {}
    for task_name in task_names:
        target_images[task_name] = []
        reference_images[task_name] = None
        print(task_name)
        for image_name in sorted(Path(args.target_folder + "/" + task_name).glob("*.jpg")):
            target_images[task_name].append(image_name)
        for image_name in sorted(Path(args.reference_folder + "/" + task_name).glob("*.jpg")):
            reference_images[task_name] = image_name
    
        # Load the DINO model
    dinov2 = DinoV2ImageProcessor()
    for task_name in task_names:
        reference_image_name = reference_images[task_name]
        first_frame = cv2.imread(str(reference_image_name))[:, :, ::-1]
        first_frame = cv2.resize(first_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
        first_frame_annotation = np.array(Image.open(str(reference_image_name).replace(".jpg", "_mask.png")))
        for target_image_name in target_images[task_name]:
            target_image_name = str(target_image_name)
            target_annotation_folder = target_image_name.replace("target_images", "sam_results").replace(".jpg", "")


            sam_output_path = target_annotation_folder
            assert(os.path.exists(sam_output_path))
            sam_output = sam_output_path
            
            sam_masks = []
            for image_name in Path(sam_output).glob("*.png"):
                binary_mask = cv2.imread(str(image_name), cv2.IMREAD_UNCHANGED)
                sam_masks.append(binary_mask)

            overall_masks = np.stack(sam_masks, axis=0)
            patch_size = 14

            feature_list = []
            images = []

            img = cv2.imread(target_image_name)

            original_img_size = img.shape[:2]
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
            print(img.shape)
            frame2_img = img
            frame1_img = cv2.resize(first_frame, (448, 448), interpolation=cv2.INTER_NEAREST)
            for img in [frame1_img, frame2_img]:
                images.append(img)
                features = dinov2.process_image(img)
                feature_list.append(features)
            h = 32
            w = 32
            aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w))

            # save the cost volume into numpy file
            cost_volume_name = os.path.join(args.output_folder, task_name, "cost_volume_{}.npy".format(target_image_name.split("/")[-1].replace(".jpg", "")))
            np.save(cost_volume_name, aff.detach().cpu().numpy())

            max_mask_id = first_frame_annotation.max()

            corresponding_mask_ids = {}
            new_annotation_mask = np.zeros_like(first_frame_annotation)
            for mask_id in range(1, max_mask_id + 1):
                instance_mask = (first_frame_annotation == mask_id).astype(np.float32)
                resized_instance_mask = cv2.resize(instance_mask, (448, 448), interpolation=cv2.INTER_NEAREST)
                print("Mask id: ", mask_id)
                print(resized_instance_mask.max())
                print(resized_instance_mask.shape)
                patchified_mask = rearrange(resized_instance_mask, '(h1 p1) (w1 p2) -> h1 w1 (p1 p2)', p1=patch_size, p2=patch_size)
                patchified_mask = np.sum(patchified_mask, axis=-1) / (patch_size * patch_size)
                new_mask = patchified_mask > 0.5
                # new_mask = patchified_mask.astype(np.uint8)
                mask_ids = np.where(new_mask == 1)

                overlapped_indices = []

                for i, j in zip(mask_ids[0], mask_ids[1]):
                    select_aff = aff[i, j]
                    h, w = select_aff.shape
                    image_flat = select_aff.reshape(h * w, 1) / 100
                    # Apply softmax
                    # softmax = np.exp(image_flat) / np.sum(np.exp(image_flat), axis=0)
                    softmax = torch.exp(image_flat) / torch.sum(torch.exp(image_flat), axis=0)
                    # Reshape softmax back to the original image shape
                    select_aff = softmax.reshape(h, w)

                    # Select top k values
                    top_k = 20
                    threshold = softmax[softmax.squeeze().argsort(descending=True)[top_k]].detach().cpu().numpy()
                    select_aff = rescale_feature_map(select_aff.unsqueeze(0).unsqueeze(0), original_img_size[0], original_img_size[1]).squeeze()

                    binary_mask = select_aff > threshold

                    iou = compute_iou(binary_mask, overall_masks)
                    overlapped_indices.append(iou.argsort()[::-1][0])
                    print(overlapped_indices)

                corresponding_mask_ids[mask_id] = find_most_repeated_number(overlapped_indices)

                resize_mask = cv2.resize(sam_masks[corresponding_mask_ids[mask_id]], first_frame_annotation.shape[:2], interpolation=cv2.INTER_NEAREST)

                new_annotation_mask[np.where(resize_mask == 255)] = mask_id
                # save the mask to the folder

            new_annotation_folder = args.output_folder + "/" + task_name
            os.makedirs(new_annotation_folder, exist_ok=True)
            print(f"Saving to {new_annotation_folder}")
            new_annotation = Image.fromarray(new_annotation_mask.astype(np.uint8))
            new_annotation.putpalette(get_palette())
            new_annotation_name = os.path.join(new_annotation_folder, "frame_annotation_{}.png".format(target_image_name.split("/")[-1].replace(".jpg", "")))
            print("Saving to ", new_annotation_name)
            new_annotation.save(new_annotation_name)

            new_first_frame = cv2.resize(img, first_frame.shape[:2], interpolation=cv2.INTER_NEAREST)
            colored_mask = np.array(new_annotation.convert("RGB"))
            overlay_img = cv2.addWeighted(new_first_frame, 0.5, colored_mask, 0.5, 0)
            image_name = os.path.join(new_annotation_folder, "frame_{}.jpg".format(target_image_name.split("/")[-1].replace(".jpg", "")))
            overlay_image_name = os.path.join(new_annotation_folder, "overlay_image_{}.png".format(target_image_name.split("/")[-1].replace(".jpg", "")))
            cv2.imwrite(image_name, new_first_frame)
            cv2.imwrite(overlay_image_name, overlay_img)


if __name__ == "__main__":
    main()
