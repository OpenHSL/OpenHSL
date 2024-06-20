from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask

from openhsl.paint.utils import cluster_hsi, ANDI, ANDVI

PATH_TO_HSI = './test_data/bortnik_3/1_corn/hsi/corn1.mat'
HSI_KEY = 'image'

PAINT_UTIL = 'Cluster'

ANDI_AREA_1 = (slice(..., ...),
               slice(..., ...))
ANDI_AREA_2 = (slice(..., ...),
               slice(..., ...))

CL_CLUSTER_TYPE = 'KMeans'
CL_NUM_CLUSTERS = 2

PATH_TO_SAVE_DIR = './test_data/results/mask'
MASK_NAME = 'andvi_mask'
MASK_EXTENSION = 'npy'
MASK_KEY = None

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

mask = HSMask()
mask.add_void_layer(0, shape=raw_mask.shape)
mask.add_completed_layer(pos=1, layer=raw_mask)
mask.save(path_to_file=PATH_TO_SAVE_MASK,
          key=MASK_KEY)
