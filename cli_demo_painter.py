from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask

from openhsl.paint.utils import cluster_hsi, ANDI, ANDVI

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь до файла ГСИ, допустимые форматы MAT, NPY, TIFF, H5
PATH_TO_HSI = '...'

# Указывается ключ файла ГСИ, если формат MAT или H5, иначе None
HSI_KEY = '...'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается тип метода автоматизированной разметки
PAINT_UTIL = 'ANDVI'

# ----------------------------------------------------------------------------------------------------------------------

# При использовании ANDI метода требуется указать координаты (верхний левый и правый нижний углы)
# двух областей, по которым требуется провести сегментацию ГСИ
ANDI_AREA_1 = (slice(..., ...),
               slice(..., ...))
ANDI_AREA_2 = (slice(..., ...),
               slice(..., ...))

# ----------------------------------------------------------------------------------------------------------------------

# Указывается тип кластеризатора 'KMeans' или 'SpectralClustering'
CL_CLUSTER_TYPE = 'KMeans'

# Указывается количество кластеров
CL_NUM_CLUSTERS = 2

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к директории куда будет сохранена маска разметки ГСИ
PATH_TO_SAVE_DIR = '...'

# Указывается имя файла маски разметки ГСИ
MASK_NAME = '...'

# Указывается формат файла маски разметки ГСИ, допустимые значения 'mat', 'npy', 'h5', 'tiff', 'png', 'bmp'
MASK_EXTENSION = '...'

# Указывается ключ файла маски разметки ГСИ, если использовались форматы 'mat' или 'h5' в MASK_EXTENSION
MASK_KEY = None

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

PATH_TO_SAVE_MASK = f'{PATH_TO_SAVE_DIR}/{MASK_NAME}.{MASK_EXTENSION}'


hsi = HSImage()
hsi.load(path_to_data=PATH_TO_HSI,
         key=HSI_KEY)

if PAINT_UTIL == 'Cluster':
    raw_mask = cluster_hsi(hsi=hsi,
                           n_clusters=CL_NUM_CLUSTERS,
                           cl_type=CL_CLUSTER_TYPE)
elif PAINT_UTIL == 'ANDVI':
    raw_mask = ANDVI(hsi=hsi)
elif PAINT_UTIL == 'ANDI':
    ex_1 = hsi.data[ANDI_AREA_1[1], ANDI_AREA_1[0], :]
    ex_2 = hsi.data[ANDI_AREA_2[1], ANDI_AREA_2[0], :]
    raw_mask = ANDI(hsi=hsi,
                    example_1=ex_1,
                    example_2=ex_2)
else:
    raise ValueError('Unsupported paint util type')

mask = HSMask(raw_mask.astype('uint8'))
mask.save(path_to_file=PATH_TO_SAVE_MASK,
          key=MASK_KEY)
