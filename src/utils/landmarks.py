def restore_landmarks(landmarks, f, margins):
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

def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
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