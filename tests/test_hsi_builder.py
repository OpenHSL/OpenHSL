import numpy as np
import pytest

from openhsl.base.hsi import HSImage
from openhsl.build.builder import HSBuilder

# Тестирование проводилось на наборах данных 1, 4 и 5, выложенных вместе с платформой


# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к видеофайлу
path_to_avi = '.../source/source_corn_1.avi'

# Указывается путь к файлу содержащему метаданные для формирования
path_to_metadata_avi = '.../source/build_metadata.json'

# Указывается путь к файлу ГСИ
path_to_avi_hsi = '.../hsi/corn1.mat'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к видеофайлу
path_to_png = '.../source/images'

# Указывается путь к файлу содержащему метаданные для формирования
path_to_metadata_png = '.../source/build_metadata.json'

# Указывается путь к файлу ГСИ
path_to_png_hsi = '.../hsi/artery.h5'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь к видеофайлу
path_to_bmp = '.../source/images'

# Указывается путь к файлу содержащему метаданные для формирования
path_to_metadata_bmp = '.../source/build_metadata.json'

# Указывается путь к файлу ГСИ
path_to_bmp_hsi = '..../hsi/vessel1.mat'

# ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def get_avi_hsi():
    hsi = HSImage()
    hsi.load(path_to_avi_hsi, key='image')
    return hsi


@pytest.fixture
def get_png_hsi():
    hsi = HSImage()
    hsi.load(path_to_png_hsi, key='image')
    return hsi


@pytest.fixture
def get_bmp_hsi():
    hsi = HSImage()
    hsi.load(path_to_bmp_hsi, key='image')
    return hsi


def test_build_from_avi(get_avi_hsi):
    hsb = HSBuilder(path_to_data=path_to_avi,
                    path_to_metadata=path_to_metadata_avi,
                    data_type='video')
    hsb.build()
    builded_hsi = hsb.get_hsi()
    builded_hsi.rot90()
    assert np.all(builded_hsi.data == get_avi_hsi.data)


def test_build_from_png(get_png_hsi):
    hsb = HSBuilder(path_to_data=path_to_png,
                    path_to_metadata=path_to_metadata_png,
                    data_type='images')
    hsb.build()
    builded_hsi = hsb.get_hsi()
    for _ in range(3):
        builded_hsi.rot90()
    assert np.all(builded_hsi.data == get_png_hsi.data)


def test_build_from_bmp(get_bmp_hsi):
    hsb = HSBuilder(path_to_data=path_to_bmp,
                    path_to_metadata=path_to_metadata_bmp,
                    data_type='images')
    hsb.build()
    builded_hsi = hsb.get_hsi()
    for _ in range(3):
        builded_hsi.rot90()
    assert np.all(builded_hsi.data == get_bmp_hsi.data)
