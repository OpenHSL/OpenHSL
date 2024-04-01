import copy
import numpy as np

from typing import Optional, Tuple

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask


def get_dataset(hsi: HSImage, mask: Optional[HSMask] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    gt : np.ndarray
        mask of hyperspectral image

    """
    ignored_labels = [0]
    if isinstance(hsi, HSImage):
        for n, ch in enumerate(hsi):
            if np.all(np.unique(ch) == [0]):
                print(f"WARNING! {n}-CHANNEL HAS NO VALID DATA! ONLY ZEROS IN DATA!")
            else:
                continue

        img = hsi.data
    elif isinstance(hsi, np.ndarray):
        img = copy.deepcopy(hsi)
    else:
        raise Exception(f"Wrong type of hsi, {type(hsi)}")

    if np.any(mask):
        if isinstance(mask, HSMask):
            gt = mask.get_2d()
        elif isinstance(mask, np.ndarray):
            gt = copy.deepcopy(mask)
        else:
            raise Exception(f"Wrong type of mask, {type(mask)}")
        gt = gt.astype("int32")
    else:
        gt = None

    return img, gt
