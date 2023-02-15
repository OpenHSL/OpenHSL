from hsi import HSImage
from hs_mask import HSMask
from models.m1dcnn import M1DCNN
from matplotlib import pyplot as plt
from Firsov_Legacy.utils import convert_to_color_

from sklearn.metrics import classification_report

import torch

print(torch.cuda.is_available())

hsi = HSImage(None, None)
mask = HSMask(None, None)

hsi.load_from_mat('test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')

cnn = M1DCNN(n_classes=mask.n_classes,
             n_bands=hsi.data.shape[-1],
             path_to_weights='checkpoints/m1_dcnn__net/m1dcnn/2023_02_15_15_20_06_epoch10_0.76.pth',
             device='cuda')

#cnn.fit(X=hsi, y=mask, train_sample_percentage=0.5, epochs=10)

pred = cnn.predict(X=hsi)

plt.imshow(pred)
plt.show()

color_pred = convert_to_color_(pred)

plt.imshow(pred > 0)
plt.show()

plt.imshow(mask.data > 0)
plt.show()

pred[mask.data == 0] = 0
print(classification_report(pred.flatten(), mask.data.flatten()))