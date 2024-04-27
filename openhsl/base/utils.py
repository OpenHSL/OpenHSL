import numpy as np

from typing import List, Literal, Tuple

from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask


def unite_hsi_and_mask(hsi_list: List[HSImage, np.ndarray],
                       mask_list: List[HSMask, np.ndarray],
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
