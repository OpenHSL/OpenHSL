from pathlib import Path
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from openhsl.data.utils import convert_to_color_, get_palette
from openhsl.hs_mask import HSMask


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


def draw_colored_mask(mask: HSMask, predicted_mask: np.array = None):

    def tmp(l: list):
        return [i / 255 for i in l]

    palette = get_palette(np.max(mask.get_2d()))

    color_gt = convert_to_color_(mask.get_2d(), palette=palette)
    t = 1
    cmap = {k: tmp(rgb) + [t] for k, rgb in palette.items()}

    patches = [mpatches.Patch(color=cmap[i], label=mask.label_class[str(i)]) for i in cmap]

    plt.figure(figsize=(12, 12))
    if np.any(predicted_mask):
        color_pred = convert_to_color_(predicted_mask, palette=palette)
        combined = np.vstack((color_gt, color_pred))
        plt.imshow(combined, label='Colored ground truth and predicted masks')
    else:
        plt.imshow(color_gt, label='Colored ground truth mask')

    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.show()
    return color_gt
