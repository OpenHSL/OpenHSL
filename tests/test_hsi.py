import pytest
from scipy.io import loadmat
from openhsl.hsi import HSImage
import numpy as np


@pytest.fixture
def return_hsi():
    return loadmat('../test_data/tr_pr/PaviaU.mat')['paviaU']


@pytest.fixture
def return_HSImage(return_hsi):
    hsi = HSImage(return_hsi)
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


def test_load_from_mat(return_hsi):
    pass


def test_load_from_npy(return_hsi):
    pass


def test_load_from_h5(return_hsi):
    pass

