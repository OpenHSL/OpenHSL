import re
import torch
import itertools
import numpy as np
import sklearn.model_selection
import seaborn as sns
from openhsl.hs_mask import HSMask
from typing import Tuple, Literal, Union
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ScaleType = Literal['band', 'pixel']
SamplerMod = Literal['random', 'fixed', 'disjoint']


def __prepare_pred_target(prediction, target):
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

    return accuracy_score(target, prediction, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_f1(prediction: np.ndarray,
           target: Union[np.ndarray, HSMask],
           *args,
           **kwargs):
    prediction, target = __prepare_pred_target(prediction, target)

    return f1_score(prediction, target, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_confusion_matrix():
    pass
# ----------------------------------------------------------------------------------------------------------------------


class HyperStandardScaler:

    def __init__(self, per: ScaleType = 'band'):
        """

        Parameters
        ----------
        per: str
            "band" or "pixel"
        """
        self.mu = 0
        self.s = 1
        self.per = per

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.per == 'band':
            return (X - self.mu[None, None, :]) / self.s[None, None, :]
        if self.per == 'pixel':
            return (X - self.mu[:, :, None]) / self.s[:, :, None]

    def fit(self, X: np.ndarray):
        if self.per == 'band':
            self.mu = np.mean(X, axis=(0, 1))
            self.s = np.std(X, axis=(0, 1))
        elif self.per == 'pixel':
            self.mu = np.mean(X, axis=2)
            self.s = np.std(X, axis=2)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
# ----------------------------------------------------------------------------------------------------------------------


class HyperMinMaxScaler:

    def __init__(self, per: ScaleType = 'band'):
        """

        Parameters
        ----------
        per: str
            "band" or "pixel"
        """
        self.min_ = 0
        self.max_ = 1
        self.per = per

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.per == 'band':
            return (X - self.min_[None, None, :]) / (self.max_[None, None, :] - self.min_[None, None, :])
        if self.per == 'pixel':
            return (X - self.min_[:, :, None]) / (self.max_[:, :, None] - self.min_[:, :, None])

    def fit(self, X: np.ndarray):
        if self.per == 'band':
            self.min_ = np.min(X, axis=(0, 1))
            self.max_ = np.max(X, axis=(0, 1))
        elif self.per == 'pixel':
            self.min_ = np.min(X, axis=2)
            self.max_ = np.max(X, axis=2)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
# ----------------------------------------------------------------------------------------------------------------------


def split_train_test_set(X: np.ndarray,
                         y: np.ndarray,
                         test_ratio: float):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
# ----------------------------------------------------------------------------------------------------------------------


def create_patches(X: np.ndarray,
                   y: np.ndarray,
                   patch_size: int = 5,
                   remove_zero_labels: bool = True):

    margin = int((patch_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
    patches_labels = np.zeros((X.shape[0] * X.shape[1]))

    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patches_labels[patch_index] = y[r - margin, c - margin]
            patch_index = patch_index + 1

    if remove_zero_labels:
        patches_data = patches_data[patches_labels > 0, :, :, :]
        patches_labels = patches_labels[patches_labels > 0]

    return patches_data, patches_labels
# ----------------------------------------------------------------------------------------------------------------------


def is_coordinate_in_padded_area(coordinates: Tuple,
                                 image_size: Tuple,
                                 padding_size: int) -> bool:
    """

    Args:
        coordinates:
        image_size:
        padding_size:

    Returns:

    """
    x, y = coordinates
    is_in_x = padding_size < x < image_size[0] - padding_size
    is_in_y = padding_size < y < image_size[1] - padding_size
    return is_in_x and is_in_y
# ----------------------------------------------------------------------------------------------------------------------


def apply_pca(X: np.ndarray,
              num_components: int = 75,
              pca=None):
    """

    Args:
        X:
        num_components:
        pca:

    Returns:

    """
    newX = np.reshape(X, (-1, X.shape[2]))
    if pca:
        print(f'Will apply PCA from {X.data.shape[-1]} to {len(pca.components_)}')
        newX = pca.transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], len(pca.components_)))
    else:
        print(f'Will apply PCA from {X.data.shape[-1]} to {num_components}')
        pca = PCA(n_components=num_components, whiten=True, random_state=131)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX, pca
# ----------------------------------------------------------------------------------------------------------------------


def pad_with_zeros(X: np.ndarray,
                   margin: int = 2):
    """

    Args:
        X:
        margin:

    Returns:

    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
# ----------------------------------------------------------------------------------------------------------------------


def standardize_input_data(data: np.ndarray) -> np.ndarray:
    """
    standardize_input_data(data)

        Transforms input data to mu=0 and std=1

        Parameters
        ----------
        data: np.ndarray

        Returns
        -------
        np.ndarray
    """
    data_new = np.zeros(np.shape(data))
    for i in range(data.shape[-1]):
        data_new[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])
    return data_new
# ----------------------------------------------------------------------------------------------------------------------


def get_device(ordinal: int):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device
# ----------------------------------------------------------------------------------------------------------------------


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h
# ----------------------------------------------------------------------------------------------------------------------


def count_sliding_window(top,
                         step=10,
                         window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)
# ----------------------------------------------------------------------------------------------------------------------


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
# ----------------------------------------------------------------------------------------------------------------------


def sample_gt(gt: np.ndarray,
              train_size: float,
              mode: SamplerMod = 'random',
              msg: str = 'train/test'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        train_size: [0, 1] float
        mode: str
        msg: str
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(X,
                                                                               train_size=train_size,
                                                                               random_state=42)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    # get fixed count of class sample for each class
    elif mode == 'fixed':
        print(f"Sampling {mode} with train size = {train_size} for {msg}")
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            train, test = sklearn.model_selection.train_test_split(X,
                                                                   train_size=train_size)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError(f"{mode} sampling is not implemented yet.")
    return train_gt, test_gt
# ----------------------------------------------------------------------------------------------------------------------


def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
# ----------------------------------------------------------------------------------------------------------------------


def get_palette(num_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return palette
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        palette = get_palette(np.max(arr_2d))

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
# ----------------------------------------------------------------------------------------------------------------------
