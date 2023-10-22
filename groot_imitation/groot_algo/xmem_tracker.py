import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import yaml
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms

from groot_imitation.groot_algo import GROOT_ROOT_PATH

from third_party.XMem.model.network import XMem
from third_party.XMem.inference.inference_core import InferenceCore
from third_party.XMem.inference.data.mask_mapper import MaskMapper
from third_party.XMem.util.palette import davis_palette


FOLDER_PATH = os.path.dirname(__file__)

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

class XMemTracker:
    """
     This is a wrapper over XMem tracker.
    """
    def __init__(self, xmem_checkpoint, device, half_mode=False) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open(os.path.join(GROOT_ROOT_PATH, "vision_model_configs/xmem_config.yaml"), 'r') as stream: 
            config = yaml.safe_load(stream) 
        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()

        self.half_mode = half_mode
        if half_mode:
            network.half()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.device = device
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        self._palette = davis_palette


    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        if self.half_mode:
            if mask is not None:
                mask = mask.half()
            frame_tensor = frame_tensor.half()
        # track one frame
        with torch.autocast(device_type="cuda"):
            probs = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()

        return final_mask
    
    @property
    def palette(self):
        return self._palette

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()

    def track_video(self, video_frames, initial_mask):
        """
        Track a series of images in a single function.
        """
        masks = []
        for (i, frame) in enumerate(video_frames):
            if i == 0:
                mask = self.track(frame, initial_mask)
            else:
                mask = self.track(frame)
            masks.append(mask)
        return masks
    
    def save_colored_masks(self, masks, output_dir):
        """_summary_

        Args:
            masks (np.array): H X W logits
        """
        for i, mask in enumerate(masks):
            colored_mask = Image.fromarray(mask)
            colored_mask.putpalette(self.palette)
            colored_mask.save(os.path.join(output_dir, f"{i:07d}.png"))