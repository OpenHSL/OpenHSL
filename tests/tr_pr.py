from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.models.baseline import BASELINE
from openhsl.models.hsicnn_luo import HSICNN
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.m1dcnn import M1DCNN
from openhsl.models.m3dcnn_sharma import M3DCNN_Sharma
from openhsl.models.m3dcnn_hamida import M3D_HAMIDA
from openhsl.models.m3dcnn_he import M3DCNN
from openhsl.models.m3dcnn_li import Li3DCNN
from openhsl.models.tf2dcnn import TF2DCNN
from openhsl.Firsov_Legacy.utils import convert_to_color_

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

import torch

print(torch.cuda.is_available())

hsi = HSImage()
mask = HSMask()

hsi.load_from_mat('../test_data/tr_pr/PaviaU.mat', mat_key='paviaU')
mask.load_mask('../test_data/tr_pr/PaviaU_gt.mat', mat_key='paviaU_gt')


cnn = NM3DCNN(n_classes=mask.n_classes,
              n_bands=hsi.data.shape[-1],
              #n_bands=30,
              #apply_pca=True,
              #path_to_weights='../tests/checkpoints/nm3_dcnn__net/m3dcnn/2023_05_25_10_30_06_epoch3_0.85.pth',
              device='cuda')

optimizer_params = {
    "learning_rate": 0.01,
    "weight_decay": 0.01
}

scheduler_params = {
    "step_size": 30,
    "gamma": 0.1
}

fit_params = {
    "epochs": 5,
    "train_split_percentage": 0.5,
    "dataloader_mode": "random",
    #"optimizer": "AdamW",
    #"optimizer_params": optimizer_params,
    #"loss": "CrossEntropyLoss",
    #"batch_size": 32,
    #"scheduler_type": 'StepLR',
    #"scheduler_params": scheduler_params
}


cnn.fit(X=hsi,
        y=mask,
        fit_params=fit_params)

plt.plot(cnn.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('losses_plot.png')
plt.show()
plt.plot(cnn.val_accs)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('val_accs.png')
plt.show()


pred = cnn.predict(X=hsi, y=mask)

plt.imshow(pred)
plt.show()

color_pred = convert_to_color_(pred)

plt.imshow(color_pred)
plt.show()

print(classification_report(pred.flatten(), mask.get_2d().flatten()))


#hsi = loadmat('../test_data/tr_pr/PaviaU.mat')['paviaU']
#gt = loadmat('../test_data/tr_pr/PaviaU_gt.mat')['paviaU_gt']

#hsi.load_from_mat('../test_data/nn_tests/three_coffee_piles_small2.mat', mat_key='image')
#mask.load_mask('../test_data/nn_tests/three_coffee_piles_small2_gt.mat', mat_key='img')

#class_count = len(mask) - 1
#print(class_count)

# tf2dcnn = TF2DCNN(n_classes=class_count,
#                  path_to_weights='./tmp/checkpoint/weights.h5')

#tf2dcnn.fit(X=hsi, y=mask, train_sample_percentage=0.25, epochs=10, batch_size=16)
#tf2dcnn.load_model('./tmp/checkpoint/weights.h5')
#pred = tf2dcnn.predict(hsi)

plt.imshow(pred)
plt.show()
