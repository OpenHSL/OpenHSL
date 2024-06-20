from openhsl.build.builder import HSBuilder

# Указывается путь до папки или видеофайла с набором кадров
PATH_TO_SOURCE_DATA = '...'

# Указывается путь до файла содержащего метаданные
PATH_TO_METADATA = '...'

# Указывается путь до файла содержащего данные GPS для учёта траектории
PATH_TO_GPS = None

# Указывается тип данных 'video' или 'images' содержащих набор кадров
DATA_TYPE = 'video'

# Указывается False или True для коррекции поворота щели
ROTATION_CORRECTION = False

# Указывается количество раз поворота ГСИ на 90 градусов в пространственных координатах
NUM_ROTATE_HSI_90 = 1

# Указывается False или True если требуется инвертировать каналы ГСИ
FLIP_WAVELENGTHS = False

# Указывается путь к директории, где будет сохранен сформированный ГСИ с сопутствующей метаинформацией
DIR_TO_SAVE_HSI = '...'

# Указывается имя ГСИ
HSI_NAME = '...'

# Указывается формат файла формируемого ГСИ, доступны 'mat', 'h5', 'tiff', 'npy'
HSI_EXTENSION = '...'

# Указывается ключ файла формируемого ГСИ при использовании форматов 'mat' или 'h5' в HSI_EXTENSION, иначе None
KEY = None

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

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
