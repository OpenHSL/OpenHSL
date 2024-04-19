import numpy as np
import os
import re
import wandb
import yaml

from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union

from openhsl.base.hs_mask import HSMask


def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
# ----------------------------------------------------------------------------------------------------------------------


def get_mean_weights(weights_list: List[OrderedDict]) -> OrderedDict:
    """
    mean_weights(weights_list)

    Returns calculated mean weights of models weights list

    Arguments
    ---------
        weights_list - list of models weights
    """
    mean_weight = dict()

    for key in weights_list[0].keys():
        mean_weight[key] = sum([w[key] for w in weights_list]) / len(weights_list)

    return OrderedDict(mean_weight)
# ----------------------------------------------------------------------------------------------------------------------


def init_wandb(path: str):
    """
    init_wandb(path)

        Initialize wandb from yaml file

        Parameters
        ----------
        path: str
            path to config-file wandb.yaml

        Returns
        -------
        wandb: wandb
    """

    if os.path.exists(path):
        with open(path, 'r') as file:
            wandb_config = yaml.safe_load(file)
    else:
        print("Warning! No wandb .yaml configuration file found, wandb is disabled for this run")
        return None

    # 1) wandb dict has api_key value
    if 'api_key' not in wandb_config['wandb']:
        print("Warning! 'api_key' is not found in wandb dict, wandb is disabled for this run")
        return None
    # 2) no value for api_key provided
    elif wandb_config['wandb']['api_key'] is None:
        print("Warning! 'api_key' is not provided, wandb is disabled for this run")
        return None
    # 3) '' value provided that leads to terminal case statement
    elif wandb_config['wandb']['api_key'] == '':
        print("Warning! 'api_key' value is empty string, wandb is disabled for this run")
        return None
    # 4) num of values is not 40
    elif len(wandb_config['wandb']['api_key']) != 40:
        print("Warning! 'api_key' value is not 40 symbols, wandb is disabled for this run")
        return None
    # 5) wrong key provided - in exception
    os.environ["WANDB_API_KEY"] = wandb_config['wandb']['api_key']

    try:
        wandb.login(key=wandb_config['wandb']['api_key'])

        wandb.init(project=wandb_config['wandb']['project'],
                   entity=wandb_config['wandb']['entity'],
                   name=wandb_config['wandb']['run'],
                   mode="online")
    except wandb.errors.UsageError as e:
        print(e)
        return None
    except wandb.errors.AuthenticationError as e:
        print(e)
        return None
    except wandb.errors.CommError as e:
        print(e)
        return None

    return wandb

# ----------------------------------------------------------------------------------------------------------------------


def init_tensorboard(path_dir='tensorboard'):
    """
    init_tensorboard(path_dir)

        Initialize Tensorboard SummaryWriter for logging

        Parameters
        ----------
        path_dir: str

        Returns
        -------
        writer: torch.utils.tensorboard.SummaryWriter
    """

    writer = SummaryWriter(log_dir=path_dir)

    return writer

# ----------------------------------------------------------------------------------------------------------------------


class EarlyStopping:
    """
        EarlyStopping class

        Attributes
        ----------
        tolerance: int
            number of epochs to wait after min has been hit
        min_delta: float
            minimum change in the monitored quantity to qualify as an improvement
        counter: int
            number of epochs since min has been hit
        early_stop: bool
            True if the training process has to be stopped

        Methods
        -------
        __call__(train_loss, validation_loss)
            call method to check if the training process has to be stopped
    """

    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
# ----------------------------------------------------------------------------------------------------------------------


def draw_fit_plots(model):
    """
    draw_fit_plots(model)

        Draws plot of train/val loss and plot of train/val accuracy after model fitting

        Parameters
        ----------
        model:
            model of neural network

    """
    x = [int(i) for i in range(1, len(model.train_loss) + 1)]

    plt.figure(figsize=(12, 8))
    plt.plot(x, model.train_loss, c='green', label="train loss")
    plt.plot(x, model.val_loss, c='blue', label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(x)
    plt.grid()
    plt.legend()
    plt.savefig('TrainVal_losses_plot.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(x, model.train_accs, c='green', label='train accuracy')
    plt.plot(x, model.val_accs, c='blue', label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(x)
    plt.grid()
    plt.legend()
    plt.savefig('TrainVal_accs.png')
    plt.show()

    if model.lrs:
        plt.figure(figsize=(12, 8))
        plt.plot(x, model.lrs, c='blue', label='Learning rate')
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.xticks(x)
        plt.grid()
        plt.legend()
        plt.savefig('Learning_rate.png')
        plt.show()
# ----------------------------------------------------------------------------------------------------------------------


def __prepare_pred_target(prediction: np.ndarray,
                          target: np.ndarray):
    """
    Remove all zeros masked pixels from prediction and target

    Parameters
    ----------
    prediction
    target

    Returns
    -------

    """
    prediction = prediction.flatten()

    if isinstance(target, HSMask):
        target = target.get_2d()
    target = target.flatten()

    # remove all pixels with zero-value mask
    indices = np.nonzero(target)
    prediction = prediction[indices]
    target = target[indices]

    return prediction, target
# ----------------------------------------------------------------------------------------------------------------------


def get_accuracy(prediction: np.ndarray,
                 target: Union[np.ndarray, HSMask],
                 *args,
                 **kwargs):
    prediction, target = __prepare_pred_target(prediction, target)

    return accuracy_score(y_true=target, y_pred=prediction, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_f1(prediction: np.ndarray,
           target: Union[np.ndarray, HSMask],
           *args,
           **kwargs):
    prediction, target = __prepare_pred_target(prediction, target)

    return f1_score(y_true=target, y_pred=prediction, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_confusion_matrix():
    # TODO realise it
    pass
# ----------------------------------------------------------------------------------------------------------------------
