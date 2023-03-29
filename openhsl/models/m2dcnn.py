from typing import Any, Optional
import numpy as np

from openhsl.models.model import Model
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self,
                 input_channels,
                 n_classes,
                 patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class M2DCNN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None):
        
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 64
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams["learning_rate"] = 0.05
        self.hyperparams['batch_size'] = 60
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'm2dcnn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = SharmaEtAl(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparams["learning_rate"], weight_decay=0.0005)
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
        
        self.model, self.losses, self.val_accs = super().fit_nn(X=X,
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
