from typing import Any, Optional
import numpy as np


from models.model import Model
from hsi import HSImage
from hs_mask import HSMask

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class HSICNN_Net(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
# ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 input_channels,
                 n_classes,
                 patch_size=3,
                 n_planes=90):
        super(HSICNN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)
# ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h
# ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HSICNN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None):
        
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 3
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams["learning_rate"] = 0.1
        self.hyperparams['batch_size'] = 100
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'hsicnn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = HSICNN_Net(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparams["learning_rate"], weight_decay=0.09)
        self.loss = nn.CrossEntropyLoss(weight=self.hyperparams["weights"])

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

        self.hyperparams.setdefault("supervision", "full")
        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)
        self.hyperparams.setdefault("mixture_augmentation", False)
        self.hyperparams["center_pixel"] = True
# ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            epochs: int = 10,
            train_sample_percentage: float = 0.5):

        self.model = super().fit_nn(X=X,
                                    y=y,
                                    hyperparams=self.hyperparams,
                                    epochs=epochs,
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    loss=self.loss,
                                    train_sample_percentage=train_sample_percentage)
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: HSImage,
                y: Optional[HSMask] = None) -> np.ndarray:
        prediction = super().predict_nn(X=X, y=y, model=self.model, hyperparams=self.hyperparams)
        return prediction
