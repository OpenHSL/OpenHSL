from pathlib import Path
import math
import numpy as np
from matplotlib import pyplot as plt


def dir_exists(path: str) -> bool:
    return Path(path).exists()
# ----------------------------------------------------------------------------------------------------------------------


def load_data(path: str,
              exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]
# ----------------------------------------------------------------------------------------------------------------------


def gaussian(length: int,
             mean: float,
             std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
# ----------------------------------------------------------------------------------------------------------------------


def draw_fit_plots(model):
    """
    draw_fit_plots(model)

        Draws plot of train/val loss and plot of train/val accuracy after model fitting
        Args:
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

