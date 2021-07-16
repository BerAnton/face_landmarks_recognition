from typing import List, Dict, Tuple, Any

import numpy as np
import cv2


class ScaleMinSideToSize(object):
    """Rescales image.
    :args:
        - sample - image to rescale
    :returns:
        - sample - rescaled image."""

    def __init__(self, size: Tuple[int, int], elem_name: str = "image") -> None:
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample: Dict) -> Dict:
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if "landmarks" in sample:
            landmarks = sample["landmarks"].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample["landmarks"] = landmarks.reshape(-1)

        return sample
