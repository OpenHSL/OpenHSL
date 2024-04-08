import datetime
import json
import math
import matplotlib.patches as mpatches
import numpy as np
import os
import wandb
import yaml

from itertools import product
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import List, Literal, Optional, Tuple, Union

from openhsl.data.utils import convert_to_color_, get_palette
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask


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


def get_cluster(cl_type):
    if cl_type == 'KMeans':
        return KMeans
    elif cl_type == 'SpectralClustering':
        return SpectralClustering
# ----------------------------------------------------------------------------------------------------------------------


def cluster_hsi(hsi: HSImage,
                n_clusters: int = 2,
                cl_type='Kmeans') -> np.ndarray:
    km = get_cluster(cl_type=cl_type)(n_clusters=n_clusters)
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


def neighbor_el(elements_list: list, element: float) -> float:
    """
    neighbor_el(elements_list, element)

        Return the closest element from list to given element

        Parameters
        ----------
        elements_list: list

        element: float

        Returns
        -------
            float
    """
    return min(elements_list, key=lambda x: abs(x - element))
# ----------------------------------------------------------------------------------------------------------------------


def get_band_numbers(w_l: int, w_data: Union[list, np.ndarray]) -> int:
    """
    get_band_numbers(w_l, w_data)

        Returns the required channel value in the hyperspectral image

        Parameters
        ----------
        w_l: int
           the desired wavelength (nm)

        w_data: list or np.ndarray
            list of wavelengths

        Returns
        ------
            int
    """

    if w_l in w_data:
        w_data = list(w_data)
        return w_data.index(w_l)
    else:
        w_data = np.array(w_data)
        delta = w_data - w_l
        abs_delta = list(map(abs, delta))
        index_new_wl = abs_delta.index(min(abs_delta))

        return index_new_wl
# ----------------------------------------------------------------------------------------------------------------------


def get_hypercube_and_wavelength(cube: Union[HSImage, np.ndarray],
                                 wave_data: Union[list, np.ndarray] = None) -> Tuple[np.ndarray, list]:
    """
    get_hypercube_and_wavelength(cube, wave_data)

        Returns hypercube and wavelengths, determines priority

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            np.ndarray, list
    """

    w_data = None

    if isinstance(cube, HSImage):
        cube_data = cube.data
        if any(cube.wavelengths):
            w_data = cube.wavelengths

    elif isinstance(cube, np.ndarray):
        cube_data = cube
    else:
        raise ValueError("Unvailable type of HSI")

    if np.any(wave_data):
        w_data = list(wave_data)

    if not any(w_data):
        raise ValueError("Not info about wavelengths")

    if type(w_data) != list:
        w_data = list(w_data)

    return cube_data, w_data
# ----------------------------------------------------------------------------------------------------------------------


def minmax_normalization(mask: np.ndarray) -> np.ndarray:
    """
    normalization(mask)

        Returns a normalized mask from 0 to 1

        Parameters
        ----------
        mask: np.ndarray
            Denormalized array
        Return
        ------
            np.ndarray
    """

    return (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
# ----------------------------------------------------------------------------------------------------------------------


def contrast_correction(rgb, gamma_thresh):
    gray_mean = np.mean(rgb, axis=2)
    un = np.unique(gray_mean)

    coord = np.where(gray_mean == un[int(len(un) * gamma_thresh - 1)])
    x, y = coord
    m = np.max(rgb[int(x[0]), int(y[0]), :])

    rgb[rgb > m] = m
    rgb = rgb / np.max(rgb)

    return rgb
# ----------------------------------------------------------------------------------------------------------------------


def simple_hsi_to_rgb(hsi: HSImage,
                      gamma_thresh: float = 0.98) -> np.ndarray:
    """
    simple_hsi_to_rgb(cube, wave_data)

        Return rgb-image from hyperspectral image

        Parameters
        ----------
        hsi: HSImage or np.ndarray
           hyperspectral image

        gamma_thresh

        Returns
        ------
            np.ndarray
    """

    cube_data = hsi.data
    w_data = hsi.wavelengths

    wl_440 = 440
    wl_550 = 550
    wl_640 = 640

    blue_band_numbers = get_band_numbers(wl_440, w_data)
    green_band_numbers = get_band_numbers(wl_550, w_data)
    red_band_numbers = get_band_numbers(wl_640, w_data)

    blue = cube_data[:, :, blue_band_numbers].astype(float)
    green = cube_data[:, :, green_band_numbers].astype(float)
    red = cube_data[:, :, red_band_numbers].astype(float)

    simple_rgb = np.dstack((red.astype(np.uint8), green.astype(np.uint8), blue.astype(np.uint8)))

    simple_rgb = contrast_correction(simple_rgb, gamma_thresh)

    return simple_rgb
# ----------------------------------------------------------------------------------------------------------------------


def xyz2srgb_exgamma(xyz: np.ndarray) -> np.ndarray:
    """
    See IEC_61966-2-1.pdf
    No gamma correction has been incorporated here, nor any clipping, so this
    transformation remains strictly linear.  Nor is there any data-checking.
    DHF 9-Feb-11
    """
    # Image dimensions
    d = xyz.shape
    r = d[0] * d[1]
    w = d[2]

    # Reshape for calculation, converting to w columns with r rows.
    xyz = np.reshape(xyz, (r, w))

    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).

    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0414],
                  [0.0557, -0.2040, 1.0570]])

    s_rgb = np.dot(xyz, m.T)

    # Reshape to recover shape of original input.
    s_rgb = np.reshape(s_rgb, d)

    return s_rgb
# ----------------------------------------------------------------------------------------------------------------------


def get_bounds_vlr(w_data: List):

    right_bound = w_data.index(neighbor_el(w_data, 720))
    left_bound = w_data.index(neighbor_el(w_data, 400))
    return left_bound, right_bound
# ----------------------------------------------------------------------------------------------------------------------


def convert_hsi_to_xyz(xyz_bar_path,
                       hsi,
                       rgb_waves):
    """
    Converting HSI to XYZ
    Parameters
    ----------
    xyz_bar_path
    hsi
    rgb_waves

    Returns
    -------

    """
    xyz_bar = loadmat(xyz_bar_path)['xyzbar']

    xyz_bar_0 = xyz_bar[:, 0]
    xyz_bar_1 = xyz_bar[:, 1]
    xyz_bar_2 = xyz_bar[:, 2]

    wl_vlr = np.linspace(400, 720, 33)

    f_0 = interp1d(wl_vlr, xyz_bar_0)
    f_1 = interp1d(wl_vlr, xyz_bar_1)
    f_2 = interp1d(wl_vlr, xyz_bar_2)

    xyz_0 = [f_0(i) for i in rgb_waves]
    xyz_1 = [f_1(i) for i in rgb_waves]
    xyz_2 = [f_2(i) for i in rgb_waves]

    xyz_bar_new = (np.array([xyz_0, xyz_1, xyz_2])).T

    r, c, w = hsi.shape
    radiances = np.reshape(hsi, (r * c, w))

    xyz = np.dot(radiances, xyz_bar_new)
    xyz = np.reshape(xyz, (r, c, 3))
    xyz = (xyz - np.min(xyz)) / (np.max(xyz) - np.min(xyz))
    return xyz
# ----------------------------------------------------------------------------------------------------------------------


def hsi_to_rgb(hsi: HSImage,
               xyz_bar_path: str = './xyzbar.mat',
               gamma_thresh: float = 0.98) -> np.ndarray:
    """
    hsi_to_rgb(cube, w_data, illumination_coef, xyzbar)

        Extracts an RGB image from an HSI image

        Parameters
        ----------
        hsi: HSImage or np.ndarray
            hyperspectral image

        xyz_bar_path: str
            path to mat file with CMF CIE 1931

        gamma_thresh: float
            coefficient for contrast correction

        Returns
        ------
            np.ndarray

    """

    hsi_data = hsi.data
    w_data = list(hsi.wavelengths)

    left_bound, right_bound = get_bounds_vlr(w_data)

    rgb_waves = w_data[left_bound: right_bound]

    new_cube = hsi_data[:, :, left_bound: right_bound]

    xyz = convert_hsi_to_xyz(xyz_bar_path=xyz_bar_path,
                             hsi=new_cube,
                             rgb_waves=rgb_waves)

    rgb = xyz2srgb_exgamma(xyz)
    rgb = minmax_normalization(rgb)

    rgb = contrast_correction(rgb, gamma_thresh)

    return rgb


# TODO rename to path_exists
def dir_exists(path: str) -> bool:
    return Path(path).exists()


def get_current_date(date_format_str="%d.%m.%Y") -> str:
    return datetime.datetime.now().strftime(date_format_str)


def get_current_time(time_format_str="%H:%M:%S") -> str:
    return datetime.datetime.now().strftime(time_format_str)


def key_exists_in_dict(dict_var: dict, key: str) -> bool:
    return key in dict_var


def get_file_complete_name(path: str) -> str:
    p = Path(path)
    return p.name


def load_data(path: str, exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]


def load_dict_from_json(path: str) -> Optional[dict]:
    data: Optional[dict] = None
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data


def save_dict_to_json(data: dict, path: str) -> None:
    with open(path, 'w') as handle:
        json.dump(data, handle, indent=4)


# TODO move into hs_image_utils
def gaussian(length: int, mean: float, std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
