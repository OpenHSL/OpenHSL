import cv2
import numpy as np
from scipy.ndimage import rotate

from openhsl.base.hsi import HSImage


PATH_TO_UAV_1 = '.../uav_1.npy'
PATH_TO_UAV_2 = '.../uav_2.npy'

DIR_TO_SAVE_HSI = '...'
HSI_NAME = 'river'
HSI_EXTENSION = 'mat'
KEY = 'image'

PATH_TO_SAVE_HSI = f'{DIR_TO_SAVE_HSI}/{HSI_NAME}.{HSI_EXTENSION}'


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


hsi_river_1 = HSImage()
hsi_river_1.load(path_to_data=PATH_TO_UAV_1)

hsi_river_2 = HSImage()
hsi_river_2.load(path_to_data=PATH_TO_UAV_2)

# -----------------------------------------------------------
hsi_river_1.data = hsi_river_1.data[3873: 5073, 50: -100, :]

for i in range(3):
    hsi_river_1.rot90()

hsi_1_resize = data_resize(hsi_river_1.data, x=550, y=645)
# -----------------------------------------------------------
hsi_river_2.data = rotate(hsi_river_2.data, angle=31)
hsi_river_2.data = hsi_river_2.data[161: -161, 344: -345, :]

for i in range(3):
    hsi_river_2.rot90()

hsi_river_2.data = hsi_river_2.data[420: 620, 430: 820, :]
hsi_2_resize = data_resize(hsi_river_2.data, x=550, y=975)
# -----------------------------------------------------------
river_merge_250_bands = np.hstack((hsi_1_resize, hsi_2_resize))[:, 100: -100, :]
river_merge_200_bands = river_merge_250_bands[:, :1000, :-50]
river_merge_200_bands_resize = data_resize(river_merge_200_bands, 780, 1420).astype("uint8")
# -----------------------------------------------------------
hsi = HSImage(hsi=river_merge_200_bands_resize, wavelengths=hsi_river_1.wavelengths)

hsi.save(path_to_data=PATH_TO_SAVE_HSI,
         key=KEY)
