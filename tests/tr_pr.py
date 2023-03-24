from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.models.m1dcnn import M1DCNN
from matplotlib import pyplot as plt
from Firsov_Legacy.utils import convert_to_color_

from sklearn.metrics import classification_report

import torch

print(torch.cuda.is_available())

hsi = HSImage()
mask = HSMask()

hsi.load_from_mat('../test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('../test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')

cnn = M1DCNN(n_classes=mask.n_classes,
             n_bands=hsi.data.shape[-1],
             #path_to_weights='checkpoints/m3_dcnn__net/m3dcnn/2023_02_15_15_30_07_epoch8_1.00.pth',
             device='cuda')

cnn.fit(X=hsi, y=mask, train_sample_percentage=0.5, epochs=3)

print(cnn.losses)
pred = cnn.predict(X=hsi, y=mask)

plt.imshow(pred)
plt.show()

color_pred = convert_to_color_(pred)

plt.imshow(color_pred)
plt.show()

print(classification_report(pred.flatten(), mask.get_2d().flatten()))
