import os
import pytest
import shutil
import numpy as np

from openhsl.base.hsi import HSImage


@pytest.fixture
def return_hsi():
    dummy_hsi = np.array([np.eye(200, 200) for _ in range(150)]).transpose((1, 2, 0)).astype(int)
    return dummy_hsi


@pytest.fixture
def return_wavelengths():
    return list(range(420, 720, 2))


@pytest.fixture
def return_HSImage(return_hsi, return_wavelengths):
    hsi = HSImage(hsi=return_hsi, wavelengths=return_wavelengths)
    return hsi


def test_get_layer(return_hsi, return_HSImage):
    hsi = return_HSImage
    assert np.all(return_hsi[:, :, 10] == hsi[10])


def test_get_hyperpixel(return_hsi, return_HSImage):
    x = 100
    y = 150
    hsi = return_HSImage
    assert np.all(return_hsi[y, x, :] == hsi.get_hyperpixel_by_coordinates(x=x, y=y))


def test_get_out_of_bound_hyperpixel(return_HSImage):
    x = 100000
    y = 100000
    hsi = return_HSImage
    with pytest.raises(IndexError):
        hsi.get_hyperpixel_by_coordinates(x=x, y=y)


def _create_dir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def _remove_dir(path_to_dir):
    shutil.rmtree(path_to_dir)


def test_save_and_load_mat(return_HSImage):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'mat'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSImage.save(path_to_data=path_to_save, key='test_key')
    loaded_HSImage = HSImage()
    loaded_HSImage.load(path_to_save, key='test_key')

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSImage.data == loaded_HSImage.data)
    assert np.all(return_HSImage.wavelengths == loaded_HSImage.wavelengths)


def test_save_and_load_npy(return_HSImage, return_hsi):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'npy'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSImage.save(path_to_data=path_to_save)
    loaded_HSImage = HSImage()
    loaded_HSImage.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSImage.data == loaded_HSImage.data)
    assert np.all(return_HSImage.wavelengths == loaded_HSImage.wavelengths)


def test_save_and_load_h5(return_HSImage, return_hsi):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'h5'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSImage.save(path_to_data=path_to_save, key='test_key')
    loaded_HSImage = HSImage()
    loaded_HSImage.load(path_to_save, key='test_key')

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSImage.data == loaded_HSImage.data)
    assert np.all(return_HSImage.wavelengths == loaded_HSImage.wavelengths)


def test_save_and_load_tiff(return_HSImage, return_hsi):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'tiff'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSImage.save(path_to_data=path_to_save)
    loaded_HSImage = HSImage()
    loaded_HSImage.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSImage.data == loaded_HSImage.data)
    assert np.all(return_HSImage.wavelengths == loaded_HSImage.wavelengths)
