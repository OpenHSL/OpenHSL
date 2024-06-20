import numpy as np
import pytest

from openhsl.base.hsi import HSImage
from openhsl.build.builder import HSBuilder


path_to_avi = '../test_data/bortnik_3/1_corn/source/source_corn_1.avi'
path_to_metadata_avi = '../test_data/bortnik_3/1_corn/source/build_metadata.json'
path_to_avi_hsi = '../test_data/bortnik_3/1_corn/hsi/corn1.mat'

path_to_png = '../test_data/bortnik_3/4_stained_micro/source/images'
path_to_metadata_png = '../test_data/bortnik_3/4_stained_micro/source/build_metadata.json'
path_to_png_hsi = '../test_data/bortnik_3/4_stained_micro/hsi/artery.h5'

path_to_bmp = '../test_data/bortnik_3/5_unstained_micro/source/images'
path_to_metadata_bmp = '../test_data/bortnik_3/5_unstained_micro/source/build_metadata.json'
path_to_bmp_hsi = '../test_data/bortnik_3/5_unstained_micro/hsi/vessel1.mat'


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
