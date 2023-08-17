import copy
from openhsl.data.utils import apply_pca

from typing import Any, Optional, Dict
import numpy as np

from openhsl.models.model import Model
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class BASELINE(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 apply_pca=False,
                 path_to_weights=None
                 ):
        self.apply_pca = apply_pca

        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 1
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'nn'

        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = Baseline(n_bands, n_classes, dropout=False)

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

        self.hyperparams.setdefault("supervision", "full")  # TODO WTF

        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)  # TODO AS PARAMS
        self.hyperparams.setdefault("mixture_augmentation", False)

        self.hyperparams["center_pixel"] = True  # TODO WTF
# ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            fit_params: Dict):

        if self.apply_pca:
            X = copy.copy(X)
            n_bands = self.hyperparams['n_bands']
            print(f'Will apply PCA from {X.data.shape[-1]} to {n_bands}')
            X.data, _ = apply_pca(X.data, self.hyperparams['n_bands'])
        else:
            print('PCA will not apply')

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('dataloader_mode', 'random')
        fit_params.setdefault('loss', nn.CrossEntropyLoss(weight=self.hyperparams["weights"]))  # TODO custom loss?
        fit_params.setdefault('batch_size', 100)
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.0001, 'weight_decay': 0.0005})
        fit_params.setdefault('optimizer',
                              optim.SGD(self.model.parameters(),
                                        lr=fit_params['optimizer_params']["learning_rate"],
                                        weight_decay=fit_params['optimizer_params']['weight_decay']))

        self.model, self.losses, self.val_accs = super().fit_nn(X=X,
                                                                y=y,
                                                                hyperparams=self.hyperparams,
                                                                model=self.model,
                                                                fit_params=fit_params)
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: HSImage,
                y: Optional[HSMask] = None) -> np.ndarray:

        if self.apply_pca:
            X = copy.copy(X)
            n_bands = self.hyperparams['n_bands']
            print(f'Will apply PCA from {X.data.shape[-1]} to {n_bands}')
            X.data, _ = apply_pca(X.data, self.hyperparams['n_bands'])
        else:
            print('PCA will not apply')
        self.hyperparams.setdefault('batch_size', 100)
        prediction = super().predict_nn(X=X,
                                        y=y,
                                        model=self.model,
                                        hyperparams=self.hyperparams)
        return prediction
    # ------------------------------------------------------------------------------------------------------------------
