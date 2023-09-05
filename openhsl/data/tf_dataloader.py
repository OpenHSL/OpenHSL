import random
import scipy
import numpy as np
import scipy.ndimage

from sklearn import preprocessing
from typing import Optional, Dict
from openhsl.data.utils import apply_pca, pad_with_zeros, split_train_test_set, create_patches

from keras.utils import np_utils


def standartize_data(X: np.ndarray):
    new_X = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(new_X)
    new_X = scaler.transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], X.shape[2]))
    return new_X, scaler
# ----------------------------------------------------------------------------------------------------------------------


def augment_data(X_train: np.ndarray):
    for i in range(int(X_train.shape[0] / 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if num == 0:
            flipped_patch = np.flipud(patch)
        if num == 1:
            flipped_patch = np.fliplr(patch)
        if num == 2:
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, no, axes=(1, 0),
                                                               reshape=False, output=None, order=3, mode='constant',
                                                               cval=0.0, prefilter=False)
        patch2 = flipped_patch
        X_train[i, :, :, :] = patch2

    return X_train
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_data(X: np.ndarray,
                    y: np.ndarray,
                    train_sample_percentage: float,
                    patch_size=5):

    X_patches, y_patches = create_patches(X, y, patch_size=patch_size)

    test_ratio = 1.0 - train_sample_percentage

    X_train, X_test, y_train, y_test = split_train_test_set(X_patches, y_patches, test_ratio)
    X_train, X_val, y_train, y_val = split_train_test_set(X_train, y_train, 0.1)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2]))
    y_train = np_utils.to_categorical(y_train)

    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[3], X_val.shape[1], X_val.shape[2]))
    y_val = np_utils.to_categorical(y_val)

    return X_train, X_val, y_train, y_val
# ----------------------------------------------------------------------------------------------------------------------


def get_data_generator(X: np.ndarray,
                       y: np.ndarray,
                       epochs: int):
    for _ in range(epochs):
        train_generator = zip(X, y)
        for (img, mask) in train_generator:
            yield img, mask
# ----------------------------------------------------------------------------------------------------------------------


def get_patch_by_indicis(data: np.array,
                         height_index: int,
                         width_index: int,
                         patch_size: int):

    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = data[height_slice, width_slice, :]

    return patch
# ----------------------------------------------------------------------------------------------------------------------


def get_test_generator(X: np.array,
                       patch_size: int):
    X = pad_with_zeros(X, patch_size // 2)
    height = X.shape[0]
    width = X.shape[1]
    for i in range(0, height - patch_size + 1):
        for j in range(0, width - patch_size + 1):
            image_patch = get_patch_by_indicis(X, i, j, patch_size)
            image_patch = image_patch.reshape(image_patch.shape[2],
                                              image_patch.shape[0],
                                              image_patch.shape[1]).astype('float32')
            yield image_patch
# ----------------------------------------------------------------------------------------------------------------------
