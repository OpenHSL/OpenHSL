from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.models.ssftt import SSFTT
from openhsl.models.fcnn import FCNN
#from openhsl.models.hsicnn_luo import HSICNN
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.hsicnn_luo import HSICNN as Luo
from openhsl.models.m1dcnn import M1DCNN
#from openhsl.models.m3dcnn_sharma import M3DCNN as Sharma
#from openhsl.models.m3dcnn_hamida import M3DCNN as Hamida
#from openhsl.models.m3dcnn_he import M3DCNN as He
from openhsl.models.m3dcnn_li import M3DCNN as Li
from openhsl.models.tf2dcnn import TF2DCNN
#from openhsl.models.spectralformer import SpectralFormer
from openhsl.data.utils import convert_to_color_
from openhsl.utils import draw_fit_plots, draw_colored_mask

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

import torch

from openhsl.data.utils import apply_pca


hsi_path = '../demo_data/corn_1.mat'
hsi_key = 'image'
mask_path = '../demo_data/mask_corn_1.mat'
mask_key = 'img'

hsi = HSImage()
mask = HSMask()

hsi.load(path_to_data=hsi_path, key=hsi_key)
mask.load(path_to_data=mask_path, key=mask_key)

hsi.data, _ = apply_pca(hsi.data, 30)


optimizer_params = {
    "learning_rate": 0.001,
    "weight_decay": 0
}

scheduler_params = {
    "step_size": 1,
    "gamma": 0.1
}

augmentation_params = {
    "flip_augmentation": False,
    "radiation_augmentation": False,
    "mixture_augmentation": False
}

fit_params = {
    "epochs": 20,
    "train_sample_percentage": 0.1,
    "dataloader_mode": "fixed",
    "get_train_mask": True,
    #"wandb_vis": True,
    #"optimizer": "AdamW",
    #"optimizer_params": optimizer_params,
    #"loss": "CrossEntropyLoss",
    "batch_size": 128,
    "scheduler_type": 'StepLR',
    "scheduler_params": scheduler_params
}

cnn = SSFTT(n_classes=mask.n_classes,
             n_bands=hsi.data.shape[-1],
             #n_bands=30,
             #apply_pca=True,
             #path_to_weights='../tests/tmp/checkpoint/weights.h5',
             #path_to_weights='../tests/checkpoints/ssftt__net/ssftt/2023_10_18_13_52_09_epoch5_0.99.pth',
             device='cuda')

cnn.fit(X=hsi,
        y=mask,
        fit_params=fit_params)

draw_fit_plots(model=cnn)

pred = cnn.predict(X=hsi,
                   y=mask)

pred = pred * (mask.get_2d() > 0)

plt.imshow(pred)
plt.show()

draw_colored_mask(mask=mask,
                  predicted_mask=pred,
                  stack_type='v')

print(classification_report(pred.flatten(), mask.get_2d().flatten()))

