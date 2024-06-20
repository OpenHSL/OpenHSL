import cv2
import numpy as np

from typing import List, Literal, Tuple

from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask


def unite_hsi_and_mask(hsi_list: List,
                       mask_list: List,
                       mode: Literal['v', 'h'] = 'v') -> Tuple[HSImage, HSMask]:
    """
    hsi_list:
        [hsi_1, hsi_2, ... hsi_N]
    mask_list:
        [mask_1, mask_2, ..., mask_N]
    mode:
        'v' for vertical stacking or 'h' for horizontal stacking

    Returns united hsi and mask sets as HSImage and HSMask pair
    """
    # TODO add processing list of HSImage or HSMask

    if all(isinstance(x, HSImage) for x in hsi_list):
        pass
    if all(isinstance(x, HSMask) for x in mask_list):
        pass

    try:
        if mode == 'v':
            united_hsi = HSImage(np.vstack(hsi_list))
            united_mask = HSMask(np.vstack(mask_list))
        elif mode == 'h':
            united_hsi = HSImage(np.hstack(hsi_list))
            united_mask = HSMask(np.hstack(mask_list))
        else:
            raise Exception('wrong unite mode')
    except Exception:
        raise Exception('Incomparable size of hsi or mask in list')

    return united_hsi, united_mask


def data_resize(data: np.ndarray,
                x: int,
                y: int) -> np.ndarray:
    """
        data_resize(data, x, y, bands)

            Parameters
            ----------
            data: np.ndarray
            2-d or 3-d np.ndarray

            x: int
                the side of the new image "X"

            y: int
                the side of the new image "Y"

            Returns
            ------
                np.ndarray
        """

    if len(data.shape) == 2:
        resized = cv2.resize(data, (y, x), interpolation=cv2.INTER_AREA).astype('uint8')

    elif len(data.shape) == 3:
        resized = np.zeros((x, y, data.shape[2]))
        for i in range(data.shape[2]):
            resized[:, :, i] = cv2.resize(data[:, :, i], (y, x), interpolation=cv2.INTER_AREA).astype('uint8')
    else:
        raise ValueError("2-d or 3-d np.ndarray")
    return resized
