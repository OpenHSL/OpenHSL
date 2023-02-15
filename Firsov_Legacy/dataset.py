import numpy as np
import seaborn as sns
from typing import Optional
from hsi import HSImage
from hs_mask import HSMask


def get_dataset(hsi: HSImage, mask: Optional[HSMask]) -> tuple[np.ndarray, np.ndarray, list, list, dict]:
    """
    return data from .mat files in tuple

    Parameters
    ----------
    hsi: HSImage
    mask: HSMask
    Returns
    ----------
    img : np.array
        hyperspectral image
    gt : np.array
        mask of hyperspectral image
    palette : dict
        palette for colorizing  predicted image
    """
    ignored_labels = [0]
    img = hsi.data

    if mask:
        gt = mask.data
        label_values = np.unique(gt)
    else:
        gt = None
        label_values = [0]

    img = img.astype("float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return img, gt, ignored_labels, label_values, palette
