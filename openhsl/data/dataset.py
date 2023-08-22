import numpy as np
from typing import Optional
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.data.utils import standardize_input_data
from typing import Tuple
import copy


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
    gt : np.ndarray
        mask of hyperspectral image

    """
    ignored_labels = [0]

    for n, ch in enumerate(hsi):
        if np.all(np.unique(ch) == [0]):
            print(f"WARNING! {n}-CHANNEL HAS NO VALID DATA! ONLY ZEROS IN DATA!")
        else:
            continue

    img = hsi.data

    if mask:
        gt = mask.get_2d()
        gt = gt.astype("int32")
        label_values = np.unique(gt)
    else:
        gt = None
        label_values = [0]

    img = standardize_input_data(img).astype('float32')
    return img, gt
