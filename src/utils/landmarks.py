from typing import List

import numpy as np


def restore_landmarks(landmarks: np.ndarray, f: int, margins: List[float]) -> np.ndarray:
    """Restores landmarks to original size.
    :args:
         - landmarks - predicted landmarks.
         - f - scale factor.
         - margins - distance from bound of cropped image.
     :returns:
         - landmarks - rescaled landmarks for original image."""
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(
    landmarks: np.ndarray, fs: np.ndarray, margins_x: np.ndarray, margins_y: np.ndarray
) -> np.ndarray:
    """Restores landmarks to original size for all images in batch.
    :args:
         - landmarks - predicted landmarks.
         - f - scale factor.
         - margins - distance from bound of cropped image.
     :returns:
         - landmarks - rescaled landmarks for original image."""
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks
