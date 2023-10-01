"""
Segmentation Correspondence Model.

Input: current image observation, and a reference image and its segmentation mask

Output: a segmentation mask for the current image observation, where the each id corresponds to the id in the reference image

"""

import torch
import torch.nn as nn
import numpy as np

from collections import Counter

from einops import rearrange

from vos_3d_algo.misc_utils import get_palette, resize_image_to_same_shape
from vos_3d_algo.sam_operator import SAMOperator
from vos_3d_algo.dino_features import DinoV2ImageProcessor, compute_affinity, rescale_feature_map, generate_video_from_affinity

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

def find_most_repeated_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else -1

class SegmentationCorrespondenceModel(nn.Module):
    def __init__(self,
                 dinov2: DinoV2ImageProcessor = None,
                 sam_operator: SAMOperator = None,
                 reference_size=(448, 448)):
        super().__init__()

        self.dinov2 = DinoV2ImageProcessor() if dinov2 is None else dinov2
        self.sam_operator = SAMOperator() if sam_operator is None else sam_operator

        self.reference_size = reference_size

    def init(self):
        self.sam_operator.init()

    def compute_cost_volume(self,
                            current_obs_image,
                            ref_image,
                            h=32,
                            w=32):
        current_obs_image = resize_image_to_same_shape(current_obs_image, ref_image)
        img_list = []
        feature_list = []
        for img in [ref_image, current_obs_image]:
            img_list.append(img)
            feature_list.append(self.dinov2.process_image(img))
        aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w))
        return aff

    def forward(self, 
                current_obs_image_input, 
                ref_image_input, 
                ref_annotation_mask_input,
                h=32,
                w=32,
                patch_size=14,
                threshold=0.5,
                temperature=100,
                topk=20):
        """_summary_

        Args:
            current_obs_image_input (_type_): _description_
            ref_image_input (_type_): _description_
            ref_annotation_mask_input (_type_): _description_
            h (int, optional): _description_. Defaults to 32.
            w (int, optional): _description_. Defaults to 32.
            patch_size (int, optional): _description_. Defaults to 14.
            threshold (float, optional): _description_. Defaults to 0.5.
            temperature (int, optional): _description_. Defaults to 100.
            topk (int, optional): _description_. Defaults to 20.

        Returns:
            np.ndarray: the new annotation
        """
        
        # 1. segmetnation mask

        current_obs_image = resize_image_to_same_shape(current_obs_image_input, reference_size=self.reference_size)
        ref_image = resize_image_to_same_shape(ref_image_input, reference_size=self.reference_size)
        ref_annotation_mask = resize_image_to_same_shape(ref_annotation_mask_input, reference_size=self.reference_size)

        mask_result_dict = self.sam_operator.segment_image(current_obs_image)
        mask_ids = mask_result_dict["merged_mask"]

        raw_data = mask_result_dict["raw_data"]
        sam_masks = self.sam_operator.get_individual_masks_from_raw(raw_data)

        # 2. compute DINOv2
        aff = self.compute_cost_volume(current_obs_image, ref_image, h, w)
    
        # 3. (TODO): compute iou to find the mask
        max_mask_id = ref_annotation_mask.max()
        corresponding_mask_ids = {}
        current_annotation_mask = np.zeros_like(ref_annotation_mask)

        for mask_id in range(1, max_mask_id):
            instance_mask = (ref_annotation_mask == mask_id).astype(np.float32)
            resized_instance_mask = resize_image_to_same_shape(instance_mask, ref_image)

            patchified_mask = rearrange(resized_instance_mask, '(h1 p1) (w1 p2) -> h1 w1 (p1 p2)', p1=patch_size, p2=patch_size)
            patchified_mask = np.sum(patchified_mask, axis=-1) / (patch_size * patch_size)

            # This is to decide the patches that are on the line of mask boundaries. We assume 0.5 occupancy is considered as in the mask.
            new_mask = patchified_mask > threshold

            mask_ids = np.where(new_mask == 1)

            overlapped_indices = []

            # mask_ids[0] is an array of x coordinate, mask_ids[1] is an array of y coordinate
            for (i, j) in zip(mask_ids[0], mask_ids[1]):
                select_aff = aff[i, j]
                select_aff_h, select_aff_w = select_aff.shape
                image_flat = select_aff.reshape(select_aff_h * select_aff_w, 1) / temperature

                softmax = torch.exp(image_flat) / torch.sum(torch.exp(image_flat), axis=0)
                select_aff = softmax.reshape(select_aff_h, select_aff_w)

                topk_threshold = softmax[softmax.squeeze().argsort(descending=True)[topk]].detach().cpu().numpy()
                select_aff = rescale_feature_map(select_aff.unsqueeze(0).unsqueeze(0), current_obs_image.shape[0], current_obs_image.shape[1]).squeeze()

                binary_mask = select_aff > topk_threshold

                iou_score = compute_iou(binary_mask, np.stack(sam_masks, axis=0))

                overlapped_indices.append(iou_score.argsort()[::-1][0])
            corresponding_mask_ids[mask_id] = find_most_repeated_number(overlapped_indices)
            new_mask_resized_as_ref_seg = resize_image_to_same_shape(sam_masks[corresponding_mask_ids[mask_id]], ref_image)

            current_annotation_mask[np.where(new_mask_resized_as_ref_seg == 255)] = mask_id

        return current_annotation_mask