from hsi import HSImage
from hs_mask import HSMask

from models.model import Model

import numpy as np
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


class M3DCNN_Net(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/

    modified by N.A. Firsov, A.V. Nikonorov
    DOI: 10.18287/2412-6179-CO-1038
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(M3DCNN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.bn_conv1 = nn.BatchNorm3d(16)
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.bn_conv2_1 = nn.BatchNorm3d(16)
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.bn_conv2_2 = nn.BatchNorm3d(16)
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.bn_conv2_3 = nn.BatchNorm3d(16)
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.bn_conv2_4 = nn.BatchNorm3d(16)
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.bn_conv3_1 = nn.BatchNorm3d(16)
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.bn_conv3_2 = nn.BatchNorm3d(16)
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.bn_conv3_3 = nn.BatchNorm3d(16)
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.bn_conv3_4 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.bn_conv4 = nn.BatchNorm3d(16)
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.bn_conv1(x)

            x2_1 = self.conv2_1(x)
            x2_1 = self.bn_conv2_1(x2_1)

            x2_2 = self.conv2_2(x)
            x2_2 = self.bn_conv2_2(x2_2)

            x2_3 = self.conv2_3(x)
            x2_3 = self.bn_conv2_3(x2_3)

            x2_4 = self.conv2_4(x)
            x2_4 = self.bn_conv2_4(x2_4)

            x = x2_1 + x2_2 + x2_3 + x2_4

            x3_1 = self.conv3_1(x)
            x3_1 = self.bn_conv3_1(x3_1)

            x3_2 = self.conv3_2(x)
            x3_2 = self.bn_conv3_2(x3_2)

            x3_3 = self.conv3_3(x)
            x3_3 = self.bn_conv3_3(x3_3)

            x3_4 = self.conv3_4(x)
            x3_4 = self.bn_conv3_4(x3_4)

            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            x = self.bn_conv4(x)

            _, t, c, w, h = x.size()
        return t * c * w * h
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):

        x = F.relu(self.bn_conv1(self.conv1(x)))

        x2_1 = self.conv2_1(x)
        x2_1 = self.bn_conv2_1(x2_1)

        x2_2 = self.conv2_2(x)
        x2_2 = self.bn_conv2_2(x2_2)

        x2_3 = self.conv2_3(x)
        x2_3 = self.bn_conv2_3(x2_3)

        x2_4 = self.conv2_4(x)
        x2_4 = self.bn_conv2_4(x2_4)

        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)

        x3_1 = self.conv3_1(x)
        x3_1 = self.bn_conv3_1(x3_1)

        x3_2 = self.conv3_2(x)
        x3_2 = self.bn_conv3_2(x3_2)

        x3_3 = self.conv3_3(x)
        x3_3 = self.bn_conv3_3(x3_3)

        x3_4 = self.conv3_4(x)
        x3_4 = self.bn_conv3_4(x3_4)

        x = x3_1 + x3_2 + x3_3 + x3_4

        x = F.relu(x)
        x = F.relu(self.bn_conv4(self.conv4(x)))

        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


class M3DCNN(Model):
    def __init__(self,
                 n_classes=3,
                 n_bands=250,
                 patch_size=7,
                 path_to_weights=None,
                 device='cpu'
                 ):
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = patch_size
        self.hyperparams['batch_size'] = 40
        self.hyperparams['learning_rate'] = 0.01
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'm3dcnn'
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = M3DCNN_Net(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        self.model = self.model.to(device)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.hyperparams['learning_rate'], weight_decay=0.01)
        self.loss = nn.CrossEntropyLoss(weight=self.hyperparams["weights"])

        epoch = self.hyperparams.setdefault("epoch", 100)

        self.hyperparams["scheduler"] = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                             factor=0.1,
                                                                             patience=epoch // 4,
                                                                             verbose=True)
        self.hyperparams.setdefault("supervision", "full")
        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)
        self.hyperparams.setdefault("mixture_augmentation", False)
        self.hyperparams["center_pixel"] = True

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            epochs: int = 5,
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

        prediction = super().predict_nn(X=X,
                                        y=y,
                                        model=self.model,
                                        hyperparams=self.hyperparams)

        return prediction
# ----------------------------------------------------------------------------------------------------------------------

