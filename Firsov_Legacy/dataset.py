import numpy as np
from typing import Optional
from hsi import HSImage
from hs_mask import HSMask
from typing import Tuple


def get_dataset(hsi: HSImage, mask: Optional[HSMask]) -> Tuple[np.ndarray, np.ndarray]:
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
    gt : np.darray
        mask of hyperspectral image

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

    return img, gt
