import sys
sys.path.insert(1, '../OpenHSL')

import os
import shutil
import pytest
import numpy as np
from openhsl.hs_builder import HSBuilder
from openhsl.hsi import HSImage

path_to_copter_data = "test_data/builder/copter"
path_to_copter_metadata = "test_data/builder/copter/gps_2021-03-30.csv"
path_to_rail_data = "test_data/builder/imgs"
path_to_rotary_data = "test_data/builder/video/rec_2022-06-06-12-24-02.avi"
not_valid_path = "incorrect_path"

def saving_correct_check(hsi):
    # check if the saving is correct for .npy
    hsi.save_to_npy(path_to_file="for_test/test.npy")
    load_hsi = HSImage()
    load_hsi.load_from_npy(path_to_file="for_test/test.npy")
    assert np.allclose(load_hsi.data, hsi.data)

    # check if the saving is correct for .h5
    hsi.save_to_h5("for_test/test.h5", h5_key='image')
    load_hsi = HSImage()
    load_hsi.load_from_h5("for_test/test.h5", "image")
    assert np.allclose(load_hsi.data, hsi.data)

    # check if the saving is correct for .mat
    hsi.save_to_mat("for_test/test.mat", mat_key='image')
    load_hsi = HSImage()
    load_hsi.load_from_mat("for_test/test.mat", "image")
    assert np.allclose(load_hsi.data, hsi.data)

    # check if the saving is correct for .png
    hsi.save_to_images("for_test/png", "png")

    # check of the saving is correct for .jpg
    hsi.save_to_images("for_test/jpg", "jpg")

    os.remove("for_test/test.npy")
    os.remove("for_test/test.h5")
    os.remove("for_test/test.mat")
    shutil.rmtree("for_test/png")
    shutil.rmtree("for_test/jpg")

@pytest.fixture
def return_rail_sample():
    hsi = HSImage()
    hsi.load_from_h5(path_to_file='for_test/sample_rail.h5', h5_key='image')
    return hsi

@pytest.fixture
def return_rotary_sample():
    hsi = HSImage()
    hsi.load_from_npy(path_to_file='for_test/sample_rotary.npy')
    return hsi

@pytest.fixture
def return_copter_sample():
    hsi = HSImage()
    hsi.load_from_npy(path_to_file='for_test/sample_copter.npy')
    return hsi


def test_normal_rail(return_rail_sample):
    hsb = HSBuilder(path_to_data=path_to_rail_data,
                    data_type="images")
    hsb.build(roi=True, norm_rotation=True, principal_slices=250, light_norm=True)
    hsi = hsb.get_hsi()

    # dimension check
    assert (300, 900, 250) == hsi.data.shape

    # pattern similarity check
    assert np.allclose(return_rail_sample.data, hsi.data)

    saving_correct_check(hsi)


def test_normal_rotary(return_rotary_sample):
    hsb = HSBuilder(path_to_data=path_to_rotary_data,
                    data_type="video")
    hsb.build(principal_slices=250)
    hsi = hsb.get_hsi()

    # dimension check
    assert (1001, 2048, 250) == hsi.data.shape

    # pattern similarity check
    assert np.allclose(return_rotary_sample.data, hsi.data)

    saving_correct_check(hsi)


def test_normal_copter(return_copter_sample):
    hsb = HSBuilder(path_to_data=path_to_copter_data,
                    path_to_metadata=path_to_copter_metadata,
                    data_type="video")
    hsb.build(principal_slices=40)
    hsi = hsb.get_hsi()

    # dimension check
    assert (1724, 1080, 40) == hsi.data.shape

    # pattern similarity check
    assert np.allclose(return_copter_sample.data, hsi.data)

    saving_correct_check(hsi)


def test_incorrect_path_to_data():
    with pytest.raises(ValueError):
        hsb = HSBuilder(path_to_data=not_valid_path, data_type="video")


def test_incorrect_data_type():
    with pytest.raises(ValueError):
        hsb = HSBuilder(path_to_data=path_to_rail_data, data_type="incorrect")


def test_incorrect_type_data():
    with pytest.raises(TypeError):
        hsb = HSBuilder(path_to_data=[path_to_rail_data], data_type="video")


def test_incorrect_type_metadata():
    with pytest.raises(TypeError):
        hsb = HSBuilder(path_to_data=path_to_copter_data,
                        path_to_metadata=[path_to_copter_metadata],
                        data_type="video")
        print(hsb.path_to_metadata)

        
def test_not_valid_data_type():
    with pytest.raises(ValueError):
        hsb = HSBuilder(path_to_data=path_to_rotary_data,
                        data_type="music")
        hsb.build(principal_slices=250)


def test_copter_without_metadata():
    hsb = HSBuilder(path_to_data=path_to_copter_data,
                    data_type="video")
    hsb.build(principal_slices=10)
    hsi = hsb.get_hsi()


# TODO: It's failed 
def test_rotary_with_metadata():
    with pytest.raises(ValueError):
        hsb = HSBuilder(path_to_data=path_to_rotary_data,
                        path_to_metadata=path_to_copter_metadata,
                        data_type="video")
        hsb.build(principal_slices=10)
        hsi = hsb.get_hsi()


# TODO: It's failed
def test_rail_with_metadata():
    with pytest.raises(ValueError):
        hsb = HSBuilder(path_to_data=path_to_rail_data,
                        path_to_metadata=path_to_copter_metadata,
                        data_type="images")
        hsb.build(principal_slices=10)
        hsi = hsb.get_hsi()


def test_all_flags_for_rail():
    hsb = HSBuilder(path_to_data=path_to_rail_data,
                    data_type="images")
    hsb.build(principal_slices=250,
              norm_rotation=True,
              barrel_dist_norm=True,
              light_norm=True,
              roi=True,
              flip_wavelengths=True)
    
    hsi = hsb.get_hsi()
    # dimension check
    assert (300, 900, 250) == hsi.data.shape





