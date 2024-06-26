import numpy as np

from openhsl.base.hsi import HSImage
from openhsl.base.utils import data_resize


# Скрипт предназначен для формирования итогового ГСИ из набора данных номер 3 "Растительный микропрепарат"
# Для его запуска требуется сначала сформировать четыре ГСИ из набора кадров

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к первому файлу ГСИ
PATH_TO_PLANT_1_1 = '.../plant_1_1.npy'

# Указывается путь ко второму файлу ГСИ
PATH_TO_PLANT_1_2 = '.../plant_1_2.npy'

# Указывается путь к третьем файлу ГСИ
PATH_TO_PLANT_2_1 = '.../plant_2_1.npy'

# Указывается путь к четвертому файлу ГСИ
PATH_TO_PLANT_2_2 = '.../plant_2_2.npy'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к директории куда будет сохранен ГСИ с сопутствующей метаинформацией
DIR_TO_SAVE_HSI = '...'

# Указывается имя итогового ГСИ
HSI_NAME = 'leaves'

# Указывается формат файла итогового ГСИ, доступны 'mat', 'tiff', 'npy', 'h5'
HSI_EXTENSION = 'tiff'

# Указывается ключ файла итогового ГСИ, если используется формат файла 'mat' или 'h5' в HSI_EXTENSION
KEY = None

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

PATH_TO_SAVE_HSI = f'{DIR_TO_SAVE_HSI}/{HSI_NAME}.{HSI_EXTENSION}'

hsi_leaf_1_1 = HSImage()
hsi_leaf_1_1.load(PATH_TO_PLANT_1_1)
hsi_leaf_1_2 = HSImage()
hsi_leaf_1_2.load(PATH_TO_PLANT_1_2)
hsi_leaf_2_1 = HSImage()
hsi_leaf_2_1.load(PATH_TO_PLANT_2_1)
hsi_leaf_2_2 = HSImage()
hsi_leaf_2_2.load(PATH_TO_PLANT_2_2)

leaf_1_mix = np.vstack((hsi_leaf_1_1.data, hsi_leaf_1_2.data))
leaf_2_mix = np.vstack((hsi_leaf_2_1.data, hsi_leaf_2_2.data))

leaves = np.hstack((leaf_1_mix, leaf_2_mix))

leaves_hw1000 = data_resize(leaves, x=1000, y=1000)

hsi = HSImage(hsi=leaves_hw1000.astype("uint8"), wavelengths=hsi_leaf_1_1.wavelengths)

hsi.save(path_to_data=PATH_TO_SAVE_HSI,
         key=KEY)
