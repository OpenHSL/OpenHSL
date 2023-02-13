import model
import model_utils
from hsi import HSImage
from hs_mask import HSMask
from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.utils import sample_gt, convert_to_color_

import numpy as np
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


def _get_model(kwargs: dict) -> tuple:
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    # We train our model by AdaGrad [18] algorithm, in which
    # the base learning rate is 0.01. In addition, we set the batch
    # as 40, weight decay as 0.01 for all the layers
    # The input of our network is the HSI 3D patch in the size of 7×7×Band
    #kwargs.setdefault("patch_size", 7)
    #kwargs.setdefault("batch_size", 40)
    #lr = kwargs.setdefault("learning_rate", 0.01)
    lr = kwargs['learning_rate']
    center_pixel = True
    model = M3DCNN_Net(n_bands, n_classes, patch_size=kwargs["patch_size"])
    # For Adagrad, we need to load the model on GPU before creating the optimizer
    model = model.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    kwargs.setdefault('scheduler', None)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs
# ----------------------------------------------------------------------------------------------------------------------


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


class M3DCNN:
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
        self.hyperparams['net_name'] = 'he'
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device

        self.model, self.optimizer, self.loss, self.hyperparams = _get_model(self.hyperparams)

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            epochs: int = 5,
            sample_percentage: float = 0.5):

        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(hsi=X, mask=y)

        self.hyperparams['epoch'] = epochs

        train_gt, _ = sample_gt(gt, sample_percentage, mode='random')
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")

        # Generate the dataset

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

    def predict(self,
                X: HSImage,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.hyperparams["test_stride"] = 1
        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(X, mask=None)

        self.model.eval()

        probabilities = model_utils.test(net=self.model,
                                         img=img,
                                         hyperparams=self.hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        color_prediction = convert_to_color_(prediction, palette)

        return gt, prediction, color_prediction
# ----------------------------------------------------------------------------------------------------------------------

