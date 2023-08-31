from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
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

hsi.load(path_to_data='../test_data/tr_pr/PaviaU.mat', key='paviaU')
mask.load(path_to_data='../test_data/tr_pr/PaviaU_gt.mat', key='paviaU_gt')



optimizer_params = {
    "learning_rate": 0.1,
    "weight_decay": 0.01
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
    "epochs": 10,
    "train_sample_percentage": 0.25,
    "dataloader_mode": "fixed",
    #"optimizer": "AdamW",
    #"optimizer_params": optimizer_params,
    #"loss": "CrossEntropyLoss",
    #"batch_size": 32,
    #"scheduler_type": 'StepLR',
    #"scheduler_params": scheduler_params
}
hsi.data = hsi.data[:, :, 10:]

cnn = SpectralFormer(n_classes=mask.n_classes,
                     #n_bands=hsi.data.shape[-1],
                     n_bands=30,
             #path_to_weights='./tmp/checkpoint/weights.h5',
                     apply_pca=True,
             #path_to_weights='../tests/checkpoints/m1_dcnn__net/m1dcnn/2023_08_23_16_42_13_epoch10_0.79.pth',
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

