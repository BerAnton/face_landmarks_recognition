from typing import List, Dict, Any

import numpy as np
import torch


class CropCenter(object):
    """Crops center of image.
    :args:
        - sample - image from dataset.
    :returns:
        - sample - cropped center of image."""

    def __init__(self, size=128, elem_name: str = "image") -> None:
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample: Dict) -> Dict:
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h : margin_h + self.size, margin_w : margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if "landmarks" in sample:
            landmarks = sample["landmarks"].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample["landmarks"] = landmarks.reshape(-1)

        return sample
