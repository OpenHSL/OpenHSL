from openhsl.build.builder import HSBuilder

PATH_TO_SOURCE_DATA = './test_data/bortnik_3/1_corn/source/source_corn_1.avi'
PATH_TO_METADATA = './test_data/bortnik_3/1_corn/source/build_metadata.json'
PATH_TO_GPS = None
DATA_TYPE = 'video'

ROTATION_CORRECTION = False

NUM_ROTATE_HSI_90 = 1
FLIP_WAVELENGTHS = False

DIR_TO_SAVE_HSI = './test_data/results'
HSI_NAME = 'corn'
HSI_EXTENSION = 'npy'
KEY = None

PATH_TO_SAVE_HSI = f'{DIR_TO_SAVE_HSI}/{HSI_NAME}.{HSI_EXTENSION}'


hsb = HSBuilder(path_to_data=PATH_TO_SOURCE_DATA,
                path_to_metadata=PATH_TO_METADATA,
                path_to_gps=PATH_TO_GPS,
                data_type=DATA_TYPE)

hsb.build(norm_rotation=ROTATION_CORRECTION)

hsi = hsb.get_hsi()

for _ in range(NUM_ROTATE_HSI_90):
    hsi.rot90()

if FLIP_WAVELENGTHS:
    hsi.flip_wavelengths()

hsi.save(path_to_data=PATH_TO_SAVE_HSI,
         key=KEY)
