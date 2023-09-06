from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.models.ssftt import SSFTT
from openhsl.models.baseline import BASELINE
from openhsl.models.hsicnn_luo import HSICNN
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.m1dcnn import M1DCNN
from openhsl.models.m3dcnn_sharma import M3DCNN
from openhsl.models.m3dcnn_hamida import M3DCNN
from openhsl.models.m3dcnn_he import M3DCNN
from openhsl.models.m3dcnn_li import M3DCNN
from openhsl.models.tf2dcnn import TF2DCNN
from openhsl.models.spectralformer import SpectralFormer
from openhsl.data.utils import convert_to_color_
from openhsl.utils import draw_fit_plots

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

import torch


print(f"{torch.cuda.is_available()=}")

hsi = HSImage()
mask = HSMask()

#hsi.load(path_to_data='../test_data/tr_pr/PaviaU.mat', key='paviaU')
#mask.load(path_to_data='../test_data/tr_pr/PaviaU_gt.mat', key='paviaU_gt')

hsi.load(path_to_data='../test_data/tr_pr/three_coffee_piles.mat', key='image')
mask.load(path_to_data='../test_data/tr_pr/three_coffee_piles_new_gt.mat', key='img')

optimizer_params = {
    "learning_rate": 0.001,
    "weight_decay": 0
}

scheduler_params = {
    "step_size": 30,
    "gamma": 0.1
}

augmentation_params = {
    "flip_augmentation": False,
    "radiation_augmentation": False,
    "mixture_augmentation": False
}

fit_params = {
    "epochs": 2,
    "train_sample_percentage": 0.8,
    "dataloader_mode": "fixed",
    #"optimizer": "AdamW",
    "optimizer_params": optimizer_params,
    #"loss": "CrossEntropyLoss",
    "batch_size": 256,
    #"scheduler_type": 'StepLR',
    #"scheduler_params": scheduler_params
}

cnn = SSFTT(n_classes=mask.n_classes,
            n_bands=30,
            apply_pca=True,
            #path_to_weights='../tests/checkpoints/ssftt__net/ssftt/2023_09_06_10_56_37_epoch10_0.99.pth',
            device='cuda')

cnn.fit(X=hsi,
        y=mask,
        fit_params=fit_params)

draw_fit_plots(model=cnn)

pred = cnn.predict(X=hsi, y=mask)

pred = pred * (mask.get_2d() > 0)

plt.imshow(pred)
plt.show()

color_pred = convert_to_color_(pred)

plt.imshow(np.hstack((color_pred, convert_to_color_(mask.get_2d()))))
plt.show()

print(classification_report(pred.flatten(), mask.get_2d().flatten()))

