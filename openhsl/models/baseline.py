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
                 path_to_weights=None):
        
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 1
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams["learning_rate"] = 0.0001
        self.hyperparams['batch_size'] = 100
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'nn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = Baseline(n_bands, n_classes, dropout=False)
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

        self.model, self.losses = super().fit_nn(X=X,
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
