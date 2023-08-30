"""
Требование к точности классификации разрабатываемых алгоритмов на
предоставленных разработчиком датасетах: от 80% до 95%.

Требование к быстродействию разрабатываемых алгоритмов на
предоставленных разработчиком датасетах, базирующихся на
пространственно-спектральных сверточных сетях(кроме трансформеров), для
предсказания на ГСИ размером 200х200х250 не более 40 секунд на
графических ускорителях RTX 3090

"""
from typing import Tuple

import pytest
from sklearn.metrics import accuracy_score
from time import time

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask

from openhsl.models.baseline import BASELINE
from openhsl.models.hsicnn_luo import HSICNN
from openhsl.models.m1dcnn import M1DCNN
from openhsl.models.m3dcnn_sharma import M3DCNN as SHARMA
from openhsl.models.m3dcnn_li import M3DCNN as LI
from openhsl.models.m3dcnn_hamida import M3DCNN as HAMIDA
from openhsl.models.m3dcnn_he import M3DCNN as HE
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.tf2dcnn import TF2DCNN


@pytest.fixture
def return_inference_test_data():
    hsi = HSImage()
    mask = HSMask()
    hsi.load(path_to_data='../test_data/nn_tests/three_coffee_piles_small2.mat', key='image')
    mask.load(path_to_data='../test_data/nn_tests/three_coffee_piles_small2_gt.mat', key='img')
    return hsi, mask


def get_classification_accuracy(pretrained_model,
                                dataset: Tuple):
    X, y = dataset
    predictions = pretrained_model.predict(X)
    return accuracy_score(predictions.flatten(), y.flatten()) > 0.8


def get_inference_time(pretrained_model,
                       dataset: Tuple):
    X, y = dataset
    start_time = time()
    pretrained_model.predict(X)
    final_time = time()
    print(final_time - start_time)
    return final_time - start_time < 40


def get_accuracy(pretrained_model,
                 dataset: Tuple):
    pass


def test_baseline(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = BASELINE(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_hsicnn_luo(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = HSICNN(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_m1dcnn(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = M1DCNN(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_m3dcnn_sharma(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = SHARMA(n_classes=n_classes, device='cuda', n_bands=30, apply_pca=True)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_m3dcnn_hamida(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = HAMIDA(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_m3dcnn_he(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = HE(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_m3dcnn_li(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = LI(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_nm3dcnn(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = NM3DCNN(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)


def test_tf2dcnn(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = TF2DCNN(n_classes=n_classes, n_bands=30, apply_pca=True)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)
