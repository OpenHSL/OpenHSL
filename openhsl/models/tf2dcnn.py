import numpy as np
import scipy
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
import copy
from scipy.io import loadmat
import scipy.ndimage
from typing import Optional, Tuple, Dict

from openhsl.data.utils import apply_pca, pad_with_zeros
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask

from random import shuffle
from skimage.transform import rotate
import h5py
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DataPreprocess:
    pass
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


def standartize_data(X: np.ndarray):
    new_X = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(new_X)
    new_X = scaler.transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], X.shape[2]))
    return new_X, scaler
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
        patches_labels -= 1

    return patches_data, patches_labels
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


class TF2DCNN:

    def __init__(self,
                 n_classes: int,
                 n_bands: int,
                 apply_pca=False,
                 path_to_weights: str = None,
                 device: str = 'cpu'):

        self.patch_size = 5
        self.n_bands = n_bands
        self.class_count = n_classes
        self.apply_pca = apply_pca
        input_shape = (self.n_bands, self.patch_size, self.patch_size)

        C1 = 3 * self.n_bands

        self.model = Sequential()

        self.model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(6 * self.n_bands, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.class_count, activation='softmax'))
        sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        if path_to_weights:
            self.model.load_weights(path_to_weights)
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            fit_params: Dict):

        if self.apply_pca:
            X = copy.copy(X)
            print(f'Will apply PCA from {X.data.shape[-1]} to {self.n_bands}')
            X.data, _ = apply_pca(X.data, self.n_bands)
        else:
            print('PCA will not apply')

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('batch_size', 32)

        X_train, X_val, y_train, y_val = preprocess_data(X=X.data,
                                                         y=y.get_2d(),
                                                         train_sample_percentage=fit_params['train_sample_percentage'],
                                                         patch_size=5)

        print(f'X_train shape: {np.shape(X_train)}, y_train shape: {np.shape(y_train)}')
        print(f'X_val shape: {np.shape(X_val)}, y_val shape: {np.shape(y_val)}')

        train_generator = get_data_generator(X=X_train,
                                             y=y_train,
                                             epochs=fit_params['epochs'])

        val_generator = get_data_generator(X=X_val,
                                           y=y_val,
                                           epochs=fit_params['epochs'])

        types = (tf.float32, tf.int32)
        shapes = ((self.n_bands, self.patch_size, self.patch_size), (self.class_count,))

        ds_train = tf.data.Dataset.from_generator(lambda: train_generator, types, shapes).batch(fit_params['batch_size'])
        ds_val = tf.data.Dataset.from_generator(lambda: val_generator, types, shapes).batch(fit_params['batch_size'])

        steps = len(y_train) / fit_params['batch_size']
        val_steps = len(y_val) / fit_params['batch_size']

        checkpoint_filepath = './tmp/checkpoint'

        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)

        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        #															   save_weights_only=True,
        #															   monitor='val_accuracy',
        #															   save_best_only=True
        #															   )

        self.model.fit(ds_train,
                       validation_data=ds_val,
                       validation_steps=val_steps,
                       validation_batch_size=fit_params['batch_size'],
                       batch_size=fit_params['batch_size'],
                       epochs=fit_params['epochs'],
                       steps_per_epoch=steps,
                       verbose=1)
        #			   callbacks=[model_checkpoint_callback])

        self.losses = []
        self.val_accs = []

        self.model.save(f'{checkpoint_filepath}/weights.h5')
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: HSImage,
                y: Optional[HSMask] = None) -> np.ndarray:

        if self.apply_pca:
            X = copy.copy(X)
            print(f'Will apply PCA from {X.data.shape[-1]} to {self.n_bands}')
            X.data, _ = apply_pca(X.data, self.n_bands)
        else:
            print('PCA will not apply')

        types = tf.float32
        shapes = (self.n_bands, self.patch_size, self.patch_size)
        X = X.data

        test_generator = get_test_generator(X, patch_size=self.patch_size)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(128)

        total = sum([1 for i in ds_test])

        test_generator = get_test_generator(X, patch_size=self.patch_size)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(128)

        prediction = self.model.predict(ds_test, steps=total)
        pr = np.argmax(prediction, axis=1)
        predicted_mask = np.reshape(pr, (X.shape[0], X.shape[1]))

        return predicted_mask + 1
    # ------------------------------------------------------------------------------------------------------------------
