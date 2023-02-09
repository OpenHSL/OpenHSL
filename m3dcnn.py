import model
import os
from Firsov_Legacy.DataLoader import DataLoader
from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.utils import sample_gt, convert_to_color_, camel_to_snake, grouper, \
                                count_sliding_window, sliding_window

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data as data
import torch.optim

import numpy as np
from tqdm import tqdm
import datetime

from hsi import HSImage
from hs_mask import HSMask


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
    model = HeEtAl_bn(n_bands, n_classes, patch_size=kwargs["patch_size"])
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


class HeEtAl_bn(nn.Module):
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
        super(HeEtAl_bn, self).__init__()
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


def create_loader(img: np.array,
                  gt: np.array,
                  hyperparams: dict,
                  shuffle: bool = False):
    dataset = DataLoader(img, gt, **hyperparams)
    return data.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)


class M3DCNN:
    def __init__(self,
                 n_classes=3,
                 n_bands=250,
                 patch_size=7,
                 path_to_weights=None,
                 device='cpu'
                 ):
        self.hyperparams = {}
        self.hyperparams['patch_size'] = patch_size
        self.hyperparams['batch_size'] = 40
        self.hyperparams['learning_rate'] = 0.01
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'he'
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device

        self.model, self.optimizer, self.loss, self.hyperparams = get_model(self.hyperparams)
        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

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
        train_loader = create_loader(img, train_gt, self.hyperparams, shuffle=True)
        val_loader = create_loader(img, val_gt, self.hyperparams)

        self.train(data_loader=train_loader,
                   epoch=epochs,
                   val_loader=val_loader,
                   device=self.hyperparams['device']
        )

    def predict(self,
                X: HSImage,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.hyperparams["test_stride"] = 1
        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(X, mask=None)

        self.model.eval()

        probabilities = self.test(img, self.hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        color_prediction = convert_to_color_(prediction, palette)

        return gt, prediction, color_prediction

    def train(self,
              data_loader: data.DataLoader,
              epoch,
              display_iter=100,
              device=torch.device("cpu"),
              val_loader=None
    ):
        """
        Training loop to optimize a network for several epochs and a specified loss

        Parameters
        ----------
            optimizer: a PyTorch optimizer
            data_loader: a PyTorch dataset loader
            epoch: int specifying the number of training epochs
            criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            device (optional): torch device to use (defaults to CPU)
            display_iter (optional): number of iterations before refreshing the
            display (False/None to switch off).
            scheduler (optional): PyTorch scheduler
            val_loader (optional): validation dataset
            supervision (optional): 'full' or 'semi'
        """
        self.model.to(device)
        save_epoch = epoch // 20 if epoch > 20 else 1
        losses = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        iter_ = 1
        val_accuracies = []

        for e in tqdm(range(1, epoch + 1)):
            self.model.train()
            avg_loss = 0.0
            for batch_idx, (data, target) in (enumerate(data_loader)):
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()

                output = self.model(data)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()

                avg_loss += loss.item()
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100): iter_ + 1])

                if display_iter and iter_ % display_iter == 0:
                    string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                    string = string.format(
                        e,
                        epoch,
                        batch_idx * len(data),
                        len(data) * len(data_loader),
                        100.0 * batch_idx / len(data_loader),
                        mean_losses[iter_],
                    )

                    tqdm.write(string)

                iter_ += 1
                del (data, target, loss, output)

            # Update the scheduler
            avg_loss /= len(data_loader)
            if val_loader is not None:
                val_acc = self.val(val_loader, device=device)
                val_accuracies.append(val_acc)
                metric = -val_acc
            else:
                metric = avg_loss

            # Save the weights
            if e % save_epoch == 0:
                self.save_model(
                    camel_to_snake(str(self.model.__class__.__name__)),
                    data_loader.dataset.name,
                    epoch=e,
                    metric=abs(metric),
                )

    def val(self,
            data_loader: data.DataLoader,
            device: torch.device):
        # TODO : fix me using metrics()
        accuracy, total = 0.0, 0.0
        ignored_labels = self.hyperparams['ignored_labels']
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total

    def test(self,
             img: np.array,
             hyperparams: dict):
        """
        Test a model on a specific image
        """
        """
            Test a model on a specific image
            """
        patch_size = hyperparams["patch_size"]
        center_pixel = hyperparams["center_pixel"]
        batch_size, device = hyperparams["batch_size"], hyperparams["device"]
        n_classes = hyperparams["n_classes"]

        kwargs = {
            "step": hyperparams["test_stride"],
            "window_size": (patch_size, patch_size),
        }
        probs = np.zeros(img.shape[:2] + (n_classes,))

        iterations = count_sliding_window(img, **kwargs) // batch_size
        for batch in tqdm(
                grouper(batch_size, sliding_window(img, **kwargs)),
                total=iterations,
                desc="Inference on the image",
        ):
            with torch.no_grad():
                if patch_size == 1:
                    data = [b[0][0, 0] for b in batch]
                    data = np.copy(data)
                    data = torch.from_numpy(data)
                else:
                    data = [b[0] for b in batch]
                    data = np.copy(data)
                    data = data.transpose(0, 3, 1, 2)
                    data = torch.from_numpy(data)
                    data = data.unsqueeze(1)

                indices = [b[1:] for b in batch]
                data = data.to(device)
                output = self.model(data)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.to("cpu")

                if patch_size == 1 or center_pixel:
                    output = output.numpy()
                else:
                    output = np.transpose(output.numpy(), (0, 2, 3, 1))
                for (x, y, w, h), out in zip(indices, output):
                    if center_pixel:
                        probs[x + w // 2, y + h // 2] += out
                    else:
                        probs[x: x + w, y: y + h] += out
        return probs

    def save_model(self,
                   model_name,
                   dataset_name,
                   **kwargs):
        model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
        """
        Using strftime in case it triggers exceptions on windows 10 system
        """
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        if isinstance(self.model, torch.nn.Module):
            filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
                **kwargs
            )
            tqdm.write("Saving neural network weights in {}".format(filename))
            torch.save(self.model.state_dict(), model_dir + filename + ".pth")
        else:
            print('Saving error')
