"""
Segmentation Correspondence Model.

Input: current image observation, and a reference image and its segmentation mask

Output: a segmentation mask for the current image observation, where the each id corresponds to the id in the reference image

"""

import torch
import torch.nn as nn
import numpy as np

from vos_3d_algo.misc_utils import get_palette
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
    def __init__(self):
        super().__init__()

        self.dinov2 = DinoV2ImageProcessor()
        self.sam_operator = SAMOperator()

    def init(self):
        self.sam_operator.init()

    def forward(self, curret_obs_image, ref_image, ref_seg_mask):

        # 1. segmetnation mask
        mask_result_dict = self.sam_operator.segment_image(curret_obs_image)
        
        # 2. compute DINOv2
        # 3. compute iou to find the mask
        # 4. form the new segmentation mask
        raise NotImplementedError