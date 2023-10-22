import cv2  # type: ignore

import argparse
import json
import os
import numpy as np
import yaml
import pprint
import torch

from functools import partial
from PIL import Image

from typing import Any, Dict, List
from easydict import EasyDict
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


from groot_imitation.groot_algo import GROOT_ROOT_PATH

class SAMOperator:
    def __init__(self, 
                 model_type="vit_b",
                 checkpoint=os.path.join(GROOT_ROOT_PATH, "../" "third_party/sam_checkpoints/sam_vit_b_01ec64.pth"),
                 sam_config_file=os.path.join(GROOT_ROOT_PATH, "vision_model_configs", "sam_config.yaml"),
                 device="cuda:0", 
                 output_mode="binary_mask",
                 half_mode=True) -> None:
        with open(sam_config_file, 'r') as stream: 
            self.config = EasyDict(yaml.safe_load(stream))

        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.sam = None
        self.output_mode = output_mode

        self.half_mode = half_mode
        
        self.autocast_dtype = torch.float32
        if self.half_mode:
            self.autocast_dtype = torch.half
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=self.autocast_dtype)


    def init(self):
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)

        self.generator = SamAutomaticMaskGenerator(self.sam, output_mode=self.output_mode, **self.config)

    def print_config(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)        


    def write_masks_to_folder(self, masks: List[Dict[str, Any]], path: str) -> None:
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
        metadata = [header]
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            filename = f"{i}.png"
            cv2.imwrite(os.path.join(path, filename), mask * 255)
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)
        metadata_path = os.path.join(path, "metadata.csv")
        with open(metadata_path, "w") as f:
            f.write("\n".join(metadata))

        return
    
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img
    
    def merge_segmentation_masks(self, arrays):
        merged_mask = np.zeros_like(arrays[0]).astype(np.uint8)
        instance_id = 1
        instance_dict = {}  # Dictionary to store instance IDs and their corresponding coordinates

        for array in arrays:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i, j]:
                        merged_mask[i, j] = instance_id
                        if instance_id not in instance_dict:
                            instance_dict[instance_id] = []
                        instance_dict[instance_id].append((i, j))
            instance_id += 1

        return merged_mask, instance_dict
    
    def save_merged_mask(self, merged_mask, filepath, verbose=True):
        # Convert the merged_mask NumPy array to a PIL Image
        mask_image = Image.fromarray(merged_mask.astype(np.uint8))
        # Save the image to the specified filepath
        mask_image.save(filepath)
        if verbose:
            print(f"Saved merged mask to {filepath}.")

    def save_overall_vis_mask(self, overall_mask, filepath, verbose=True):
        cv2.imwrite(filepath, (overall_mask * 255).astype(np.uint8))
        if verbose:
            print(f"Saved overall mask to {filepath}.")
    
    def get_individual_masks_from_raw(self, raw_data):
        masks = []
        for i, mask_data in enumerate(raw_data):
            mask = mask_data["segmentation"]
            masks.append(mask * 255)
        return masks

    def segment_image(self, 
                      image,
                      merge_masks=True,
                      overall_vis_mask=True,
                      ):
        with self.autocast_ctx():
            print(self.generator, self.generator.generate)
            masks = self.generator.generate(image)
            if merge_masks:
                merged_mask, _ = self.merge_segmentation_masks([m["segmentation"] for m in masks])

            if overall_vis_mask:
                overall_mask = self.show_anns(masks)

            return {
                "raw_data": masks,
                "merged_mask": merged_mask,
                "overall_mask": overall_mask,
            }
    
    def segment_images_from_a_folder(self, 
                                     input_folder,
                                     output_folder,
                                     individual_masks=True,
                                     merge_masks=True,
                                     overall_vis_mask=True,
                                     save_masks=True,
                                     verbose=True):
        """sequentially segment images from a folder

        Args:
            input_folder (str): make sure the input folder path is configured correctly so that it is relative to the place you are running the script.
            output_folder (str): ake sure the output folder path is configured correctly so that it is relative to the place you are running the script.
            merge_masks (bool, optional): _description_. Defaults to True.
            overall_vis_mask (bool, optional): _description_. Defaults to True.
            save_masks (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """
        targets = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]

        mask_results = []
        for t in targets:
            image_name = t.split("/")[-1].split(".")[0]
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_result_dict = self.segment_image(image, 
                                                  merge_masks=merge_masks, overall_vis_mask=overall_vis_mask)
            
            if merge_masks and save_masks:
                self.save_overall_vis_mask(mask_result_dict["overall_mask"], os.path.join(output_folder, f"{image_name}_vis.png"), verbose=verbose)
            if overall_vis_mask and save_masks:
                self.save_merged_mask(mask_result_dict["merged_mask"], os.path.join(output_folder, image_name), verbose=verbose)        
            if individual_masks and save_masks:
                base = os.path.basename(t)
                base = os.path.splitext(base)[0]
                save_base = os.path.join(output_folder, base)
                self.write_masks_to_folder(mask_result_dict["masks"], save_base)
            mask_results.append(mask_result_dict)
        if verbose:
            print("SAM operator Done")
        return mask_results