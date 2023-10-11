from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.models.ssftt import SSFTT
from openhsl.models.fcnn import FCNN
#from openhsl.models.hsicnn_luo import HSICNN
#from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.hsicnn_luo import HSICNN as Luo
from openhsl.models.m1dcnn import M1DCNN
#from openhsl.models.m3dcnn_sharma import M3DCNN as Sharma
#from openhsl.models.m3dcnn_hamida import M3DCNN as Hamida
#from openhsl.models.m3dcnn_he import M3DCNN as He
from openhsl.models.m3dcnn_li import M3DCNN as Li
from openhsl.models.tf2dcnn import TF2DCNN
#from openhsl.models.spectralformer import SpectralFormer
from openhsl.data.utils import convert_to_color_
from openhsl.utils import draw_fit_plots

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

import torch


print(f"{torch.cuda.is_available()=}")


hsi_path = '../test_data/tr_pr/PaviaU.mat'
hsi_key = 'paviaU'
mask_path = '../test_data/tr_pr/PaviaU_gt.mat'
mask_key = 'paviaU_gt'

hsi = HSImage()
mask = HSMask()

#hsi_path = '../test_data/tr_pr/corn_1.mat'
#hsi_key = 'image'
#mask_path = '../test_data/tr_pr/mask_corn_1.mat'
#mask_key = 'img'


hsi.load(path_to_data=hsi_path, key=hsi_key)
mask.load(path_to_data=mask_path, key=mask_key)


print(mask.data.shape)

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
    "epochs": 5,
    "train_sample_percentage": 0.1,
    "dataloader_mode": "fixed",
    "get_train_mask": True,
    #"optimizer": "AdamW",
    #"optimizer_params": optimizer_params,
    #"loss": "CrossEntropyLoss",
    "batch_size": 32,
    #"scheduler_type": 'StepLR',
    #"scheduler_params": scheduler_params
}

cnn = SSFTT(n_classes=mask.n_classes,
          #n_bands=hsi.data.shape[-1],
          n_bands=30,
          apply_pca=True,
          #path_to_weights='../tests/tmp/checkpoint/weights.h5',
          #path_to_weights='../tests/checkpoints/li3_dcnn__net/m3dcnn_li/2023_10_09_16_45_39_epoch10_0.98.pth',
          device='cuda')

cnn.fit(X=hsi,
        y=mask,
        fit_params=fit_params)

draw_fit_plots(model=cnn)

#cnn.model.load_weights('../tests/tmp/checkpoint/weights.h5')

pred = cnn.predict(X=hsi, y=mask)

pred = pred * (mask.get_2d() > 0)

plt.imshow(pred)
plt.show()

color_pred = convert_to_color_(pred)

plt.imshow(np.hstack((color_pred, convert_to_color_(mask.get_2d()))))
plt.show()

print(classification_report(pred.flatten(), mask.get_2d().flatten()))
