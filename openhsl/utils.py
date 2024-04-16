import matplotlib.patches as mpatches
import numpy as np

from itertools import product
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from tqdm import trange
from typing import Tuple, Literal

from openhsl.data.utils import convert_to_color_, get_palette
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask


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


def draw_colored_mask(mask: HSMask,
                      predicted_mask: np.array = None,
                      mask_labels: dict = None,
                      stack_type: Literal['v', 'h'] = 'v'):

    tmp = lambda x: [i / 255 for i in x]

    palette = get_palette(np.max(mask.get_2d()))

    color_gt = convert_to_color_(mask.get_2d(), palette=palette)
    t = 1
    cmap = {k: tmp(rgb) + [t] for k, rgb in palette.items()}

    if mask_labels:
        labels = mask_labels
    else:
        labels = mask.label_class

    patches = [mpatches.Patch(color=cmap[i], label=labels.get(str(i), 'no information')) for i in cmap]

    plt.figure(figsize=(12, 12))
    if np.any(predicted_mask):
        color_pred = convert_to_color_(predicted_mask, palette=palette)
        if stack_type == 'v':
            combined = np.vstack((color_gt, color_pred))
        elif stack_type == 'h':
            combined = np.hstack((color_gt, color_pred))
        else:
            raise Exception(f'{stack_type} is unresolved mode')
        plt.imshow(combined, label='Colored ground truth and predicted masks')
    else:
        plt.imshow(color_gt, label='Colored ground truth mask')
    if labels:
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.show()

    return color_gt
# ----------------------------------------------------------------------------------------------------------------------


def __get_cluster(cl_type):
    if cl_type == 'KMeans':
        return KMeans
    elif cl_type == 'SpectralClustering':
        return SpectralClustering
# ----------------------------------------------------------------------------------------------------------------------


def cluster_hsi(hsi: HSImage,
                n_clusters: int = 2,
                cl_type='Kmeans') -> np.ndarray:
    km = __get_cluster(cl_type=cl_type)(n_clusters=n_clusters)
    h, w, _ = hsi.data.shape
    pred = km.fit_predict(hsi.to_spectral_list())
    return pred.reshape((h, w))
# ----------------------------------------------------------------------------------------------------------------------


def ANDVI(hsi: HSImage):
    def ndi(img: np.ndarray,
            l_red: int,
            r_red: int,
            l_nir: int,
            r_nir: int):
        red = np.mean(img[:, :, l_red: r_red], axis=2)
        nir = np.mean(img[:, :, l_nir: r_nir], axis=2)
        ndi_mask = (nir - red) / (nir + red)
        ndi_mask[nir + red == 0] = 0
        return ndi_mask > 0.1

    def get_ttest(img: np.ndarray,
                  plant_mask: np.ndarray):
        plant = img[plant_mask == 1]
        soil = img[plant_mask == 0]
        return ttest_ind(plant[: 1000], soil[: 1000])

    p_v = []
    for i in trange(98, 148):
        mask = ndi(hsi.data, 97, i, 148, 250)
        p_v.append(np.mean(get_ttest(hsi.data, mask)[1]))
    res_red_right = int(np.argmin(np.log(p_v)))
    print(f'right border of red is {res_red_right + 97}\'s band')
    return ndi(hsi.data, 97, res_red_right + 97, 148, 250)
# ----------------------------------------------------------------------------------------------------------------------


def norm_diff_index(channel_1: np.ndarray,
                    channel_2: np.ndarray) -> np.ndarray:
    mask = (channel_1 - channel_2) / (channel_1 + channel_2)
    mask[np.isnan(mask)] = 1
    magic_threshold = 2
    mask[mask < magic_threshold] = 0
    mask[mask >= magic_threshold] = 1
    return mask
# ----------------------------------------------------------------------------------------------------------------------


def ANDI(hsi: HSImage,
         example_1: np.ndarray,
         example_2: np.ndarray) -> np.ndarray:
    example_1_size = example_1[:, :, 0].size
    example_2_size = example_2[:, :, 0].size

    min_score = 2.0
    best_idx = None, None

    if example_1.shape[2] == example_2.shape[2]:
        channels_count = example_1.shape[2]
    else:
        raise ValueError("Different numbers of channels in two sets")

    for ind_1, ind_2 in product(range(channels_count), range(channels_count)):

        # skip main diagonal
        if ind_1 == ind_2:
            continue

        ndi = norm_diff_index(channel_1=example_1[:, :, ind_1],
                              channel_2=example_1[:, :, ind_2])

        score = np.sum(ndi) / example_1_size

        ndi = norm_diff_index(channel_1=example_2[:, :, ind_1],
                              channel_2=example_2[:, :, ind_2])

        score += (example_2_size - np.sum(ndi)) / example_2_size

        if score < min_score:
            min_score = score
            best_idx = ind_1, ind_2

    print(best_idx)

    ndi = norm_diff_index(channel_1=hsi.data[:, :, best_idx[0]],
                          channel_2=hsi.data[:, :, best_idx[1]])

    return ndi
# ----------------------------------------------------------------------------------------------------------------------


def unite_hsi_and_mask(hsi_list,
                       mask_list,
                       mode: Literal['v', 'h'] = 'v') -> Tuple[HSImage, HSMask]:
    """
    hsi_list:
        [hsi_1, hsi_2, ... hsi_N]
    mask_list:
        [mask_1, mask_2, ..., mask_N]
    mode:
        'v' for vertical stacking or 'h' for horizontal stacking

    Returns united hsi and mask sets as HSImage and HSMask pair
    """
    try:
        if mode == 'v':
            united_hsi = HSImage(np.vstack(hsi_list))
            united_mask = HSMask(np.vstack(mask_list))
        elif mode == 'h':
            united_hsi = HSImage(np.hstack(hsi_list))
            united_mask = HSMask(np.hstack(mask_list))
        else:
            raise Exception('wrong unite mode')
    except Exception:
        raise Exception('Incomparable size of hsi or mask in list')

    return united_hsi, united_mask
