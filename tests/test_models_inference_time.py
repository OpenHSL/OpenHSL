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
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from time import time

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.data.utils import apply_pca, HyperStandardScaler

from openhsl.models.fcnn import FCNN
from openhsl.models.hsicnn_luo import HSICNN as Luo
from openhsl.models.m1dcnn import M1DCNN
from openhsl.models.m3dcnn_sharma import M3DCNN as SHARMA
from openhsl.models.m3dcnn_li import M3DCNN as LI
from openhsl.models.m3dcnn_hamida import M3DCNN as HAMIDA
from openhsl.models.m3dcnn_he import M3DCNN as HE
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.tf2dcnn import TF2DCNN
from openhsl.models.ssftt import SSFTT


@pytest.fixture
def return_inference_test_data():
    hsi = HSImage(np.random.rand(200, 200, 250))
    mask = HSMask(np.random.randint(5, size=(200, 200)))
    scaler = HyperStandardScaler()
    hsi.data = scaler.fit_transform(hsi.data)
    return hsi, mask


def get_inference_time(pretrained_model,
                       dataset: Tuple,
                       num_components=None):
    X, y = dataset
    if num_components:
        X.data, _ = apply_pca(X.data, num_components=num_components)
    start_time = time()
    pretrained_model.predict(X)
    final_time = time()
    print(final_time - start_time)
    return final_time - start_time < 40


def test_m1dcnn(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = M1DCNN(n_classes=n_classes, device='cuda', n_bands=250)

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

    model = TF2DCNN(n_classes=n_classes, n_bands=30)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data, num_components=30)


def test_ssftt(return_inference_test_data):
    n_classes = len(return_inference_test_data[1])

    model = SSFTT(n_classes=n_classes, device='cuda', n_bands=250)

    assert get_inference_time(pretrained_model=model, dataset=return_inference_test_data)