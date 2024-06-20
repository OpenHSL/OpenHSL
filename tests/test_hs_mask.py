import os
import pytest
import shutil
import numpy as np

from matplotlib import pyplot as plt

from openhsl.base.hs_mask import HSMask


@pytest.fixture
def return_mask():
    first_layer = np.eye(300, 200).astype(int)
    second_layer = np.eye(300, 200)[:, ::-1].astype(int)
    zero_layer = ((first_layer + second_layer == 0) * 1).astype(int)
    dummy_mask = np.array([zero_layer, first_layer, second_layer]).transpose((1, 2, 0))
    return dummy_mask.astype(int)


@pytest.fixture
def return_class_labels():
    return {
        '0': 'void',
        '1': 'first_class',
        '2': 'second_class'
    }


@pytest.fixture
def return_HSMask(return_mask, return_class_labels):
    hsi = HSMask(mask=return_mask, label_class=return_class_labels)
    return hsi


def test_get_layer(return_mask, return_HSMask):
    assert np.all(return_mask[:, :, 0] == return_HSMask[0])
    assert np.all(return_mask[:, :, 2] == return_HSMask[2])


def _create_dir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def _remove_dir(path_to_dir):
    shutil.rmtree(path_to_dir)


def test_save_and_load_mat(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'mat'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save, key='test_key')
    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save, key='test_key')

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)


def test_save_and_load_npy(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'npy'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save)

    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)


def test_save_and_load_h5(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'h5'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save, key='test_key')
    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save, key='test_key')

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)


def test_save_and_load_tiff(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'tiff'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save)
    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)


def test_save_and_load_png(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'png'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save)
    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)


def test_save_and_load_bmp(return_HSMask):
    path_to_test_dir = './test_dir'
    _create_dir(path_to_test_dir)

    extension = 'bmp'
    path_to_save = f'{path_to_test_dir}/test_.{extension}'

    return_HSMask.save(path_to_file=path_to_save)
    loaded_HSMask = HSMask()
    loaded_HSMask.load(path_to_save)

    _remove_dir(path_to_test_dir)

    assert np.all(return_HSMask.data == loaded_HSMask.data)
    assert np.all(return_HSMask.label_class == loaded_HSMask.label_class)
