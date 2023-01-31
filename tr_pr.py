from Firsov_Legacy.utils import get_device
from Firsov_Legacy.train_model import train_model
from Firsov_Legacy.test_model import test_model
from Firsov_Legacy.utils import convert_to_color_

from hsi import HSImage
from hs_mask import HSMask
from m3dcnn import M3DCNN
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report as score
from scipy.io import loadmat

import torch

print(torch.cuda.is_available())

#WEIGHTS_PATH: str = 'checkpoints/short_he/he/2022_05_05_14_04_27_epoch15_0.96.pth' # путь до файла с весами (опционально)
SAMPLE_PERCENTAGE: float = 0.1 # размер тренировочной выборки из куба
CUDA_DEVICE = get_device(0) # подключение к доступному GPU, иначе подключается CPU

# Указываем количество эпох, классов и устройство для вычисления
hyperparams = {
        'device': CUDA_DEVICE
    }

hsi = HSImage(None, None)
mask = HSMask(None, None)

hsi.load_from_mat('test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')


cnn = M3DCNN()
cnn.fit(X=hsi, y=mask, epochs=3, hyperparams=hyperparams)

cnn.predict(X=hsi, hyperparams=hyperparams)
