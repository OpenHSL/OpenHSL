from models.model import Model

import numpy as np
import math
from typing import Any
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

from models import model_utils
from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.utils import sample_gt, convert_to_color_


class M1DCNN_Net(nn.Module):
    """
        Deep Convolutional Neural Networks for Hyperspectral Image Classification
        Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
        Journal of Sensors, Volume 2015 (2015)
        https://www.hindawi.com/journals/js/2015/258619/
        """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(M1DCNN_Net, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


class M1DCNN(Model):
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
        self.hyperparams["learning_rate"] = 0.01
        self.hyperparams['batch_size'] = 100
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'm1dcnn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = M1DCNN_Net(n_bands, n_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.hyperparams["learning_rate"])
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
            X,
            y,
            epochs,
            train_sample_percentage=0.5):
        img, gt, ignored_labels, label_values, palette = get_dataset(hsi=X, mask=y)

        self.hyperparams['epoch'] = epochs

        train_gt, _ = sample_gt(gt, train_sample_percentage, mode='random')
        train_gt, val_gt = sample_gt(train_gt, 0.9, mode="random")

        print(f'Full size: {np.sum(gt > 0)}')
        print(f'Train size: {np.sum(train_gt > 0)}')
        print(f'Val size: {np.sum(val_gt > 0)}')

        train_loader = model_utils.create_loader(img, train_gt, self.hyperparams, shuffle=True)
        val_loader = model_utils.create_loader(img, val_gt, self.hyperparams)

        self.model = model_utils.train(net=self.model,
                                       optimizer=self.optimizer,
                                       criterion=self.loss,
                                       data_loader=train_loader,
                                       epoch=epochs,
                                       val_loader=val_loader,
                                       device=self.hyperparams['device'])
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        self.hyperparams["test_stride"] = 1
        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(X, mask=None)

        self.model.eval()

        probabilities = model_utils.test(net=self.model,
                                         img=img,
                                         hyperparams=self.hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        color_prediction = convert_to_color_(prediction, palette)

        return prediction, color_prediction
# ----------------------------------------------------------------------------------------------------------------------
