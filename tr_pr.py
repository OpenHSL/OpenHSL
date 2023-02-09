from hsi import HSImage
from hs_mask import HSMask
from m3dcnn import M3DCNN
import numpy as np
from matplotlib import pyplot as plt

import torch

print(torch.cuda.is_available())


hsi = HSImage(None, None)
mask = HSMask(None, None)

hsi.load_from_mat('test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')

cnn = M3DCNN(n_classes=len(np.unique(mask.data)),
             n_bands=hsi.data.shape[-1],
             # path_to_weights='checkpoints/he_et_al_bn/he/2023_02_01_17_41_49_epoch1_0.98.pth',
             device='cpu')

cnn.fit(X=hsi, y=mask, epochs=1)

gt, pred, color_pred = cnn.predict(X=hsi)

plt.imshow(pred)
plt.show()
