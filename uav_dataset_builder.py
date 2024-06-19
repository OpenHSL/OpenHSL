import numpy as np
from scipy.ndimage import rotate

from openhsl.base.hsi import HSImage
from openhsl.base.utils import data_resize

# Скрипт предназначен для генерации ГСИ из открытого набора данных номер 10
# https://www.kaggle.com/datasets/openhsl/hyperdataset-uav-river


# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к первому ГСИ
PATH_TO_UAV_1 = '.../uav_1.npy'

# Указывается путь ко второму ГСИ
PATH_TO_UAV_2 = '.../uav_2.npy'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается директория для сохранения итогового ГСИ
DIR_TO_SAVE_HSI = '...'

# Указывается название итогового ГСИ
HSI_NAME = 'river'

# Указывается формат файла итогового ГСИ
HSI_EXTENSION = 'mat'

# Указывается ключ файла итогового ГСИ, если формат файла 'mat' или 'h5', или None
KEY = 'image'

# ----------------------------------------------------------------------------------------------------------------------

PATH_TO_SAVE_HSI = f'{DIR_TO_SAVE_HSI}/{HSI_NAME}.{HSI_EXTENSION}'

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
