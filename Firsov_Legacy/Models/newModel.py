import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init



def get_model(kwargs: dict) -> tuple:
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
    model = shortHe(n_bands, n_classes, patch_size=kwargs["patch_size"])
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

class shortHe(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(shortHe, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (10, 3, 3), stride=(3, 1, 1))
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

        self.conv4 = nn.Conv3d(16, 16, (1, 2, 2))
        self.bn_conv4 = nn.BatchNorm3d(16)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

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
