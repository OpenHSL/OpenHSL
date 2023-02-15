from hsi import HSImage
from hs_mask import HSMask
from models.m1dcnn import M1DCNN
import numpy as np
from matplotlib import pyplot as plt

import torch

print(torch.cuda.is_available())

hsi = HSImage(None, None)
mask = HSMask(None, None)

hsi.load_from_mat('test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')

cnn = M1DCNN(n_classes=mask.n_classes,
             n_bands=hsi.data.shape[-1],
             #path_to_weights='checkpoints/m3_dcnn__net/m3dcnn/2023_02_15_11_30_49_epoch1_0.98.pth',
             device='cuda')

cnn.fit(X=hsi, y=mask, train_sample_percentage=0.5, epochs=10)

pred, color_pred = cnn.predict(X=hsi)

plt.imshow(pred)
plt.show()
