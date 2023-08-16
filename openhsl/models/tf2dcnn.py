import numpy as np
import scipy
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

from scipy.io import loadmat
import scipy.ndimage
from typing import Optional, Tuple, Dict

from openhsl.utils import applyPCA, padWithZeros
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


def splitTrainTestSet(X: np.ndarray,
                      y: np.ndarray,
                      testRatio: float):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
# ----------------------------------------------------------------------------------------------------------------------


def standartizeData(X: np.ndarray):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX, scaler
# ----------------------------------------------------------------------------------------------------------------------


def createPatches(X: np.ndarray,
                  y: np.ndarray,
                  windowSize: int = 5,
                  removeZeroLabels: bool = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels
# ----------------------------------------------------------------------------------------------------------------------


def AugmentData(X_train: np.ndarray):
    for i in range(int(X_train.shape[0] / 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
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
                    numPCAcomponents=30,
                    windowSize=5):
    X, pca = applyPCA(X, numPCAcomponents)
    XPatches, yPatches = createPatches(X, y, windowSize=windowSize)

    testRatio = 1.0 - train_sample_percentage

    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)

    X_train, X_val, y_train, y_val = splitTrainTestSet(X_train, y_train, 0.1)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2]))
    y_train = np_utils.to_categorical(y_train)

    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[3], X_val.shape[1], X_val.shape[2]))
    y_val = np_utils.to_categorical(y_val)

    return X_train, X_val, y_train, y_val
# ----------------------------------------------------------------------------------------------------------------------


def get_data_generator(X: np.ndarray,
                       y: np.ndarray,
                       epochs: int):
    # image_gen = tf.data.Dataset.from_tensor_slices(X)
    # mask_gen = tf.data.Dataset.from_tensor_slices(y)
    for _ in range(epochs):
        train_generator = zip(X, y)
        for (img, mask) in train_generator:
            # img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
            # mask = np.reshape(mask, (1, mask.shape[0]))
            yield img, mask
# ----------------------------------------------------------------------------------------------------------------------


def Patch(data: np.array,
          height_index: int,
          width_index: int,
          patch_size: int):
    # transpose_array = data.transpose((2,0,1))
    # print transpose_array.shape

    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = data[height_slice, width_slice, :]

    return patch
# ----------------------------------------------------------------------------------------------------------------------


def get_test_generator(X: np.array,
                       windowSize: int,
                       numPCAcomponents: int):
    height = X.shape[0]
    width = X.shape[1]
    PATCH_SIZE = windowSize
    X, pca = applyPCA(X, numPCAcomponents)
    print(np.shape(X))
    X = padWithZeros(X)
    print(np.shape(X))
    for i in range(0, height - PATCH_SIZE):
        for j in range(0, width - PATCH_SIZE):
            image_patch = Patch(X, i, j, PATCH_SIZE)
            image_patch = image_patch.reshape(image_patch.shape[2],
                                              image_patch.shape[0],
                                              image_patch.shape[1]).astype('float32')
            yield image_patch
# ----------------------------------------------------------------------------------------------------------------------


class TF2DCNN:

    def __init__(self,
                 n_classes: int,
                 path_to_weights: str = None):

        self.windowSize = 5
        self.numPCAcomponents = 30
        self.class_count = n_classes

        input_shape = (self.numPCAcomponents, self.windowSize, self.windowSize)

        C1 = 3 * self.numPCAcomponents

        self.model = Sequential()

        self.model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(6 * self.numPCAcomponents, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.class_count, activation='softmax'))
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        if path_to_weights:
            self.model.load_weights(path_to_weights)
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            fit_params: Dict):

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('batch_size', 32)


        X_train, X_val, y_train, y_val = preprocess_data(X=X.data,
                                                         y=y.get_2d(),
                                                         train_sample_percentage=fit_params['train_sample_percentage'],
                                                         windowSize=5)

        print(f'X_train shape: {np.shape(X_train)}, y_train shape: {np.shape(y_train)}')
        print(f'X_val shape: {np.shape(X_val)}, y_val shape: {np.shape(y_val)}')

        train_generator = get_data_generator(X=X_train,
                                             y=y_train,
                                             epochs=fit_params['epochs'])

        val_generator = get_data_generator(X=X_val,
                                           y=y_val,
                                           epochs=fit_params['epochs'])

        types = (tf.float32, tf.int32)
        shapes = ((self.numPCAcomponents, self.windowSize, self.windowSize), (self.class_count,))

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

        self.model.save(f'{checkpoint_filepath}/weights.h5')
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: HSImage,
                y: Optional[HSMask] = None) -> np.ndarray:
        types = (tf.float32)
        shapes = ((self.numPCAcomponents, self.windowSize, self.windowSize))
        X = X.data

        test_generator = get_test_generator(X, windowSize=self.windowSize, numPCAcomponents=self.numPCAcomponents)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(32)

        total = (X.shape[0] - self.windowSize + 1) * (X.shape[1] - self.windowSize + 1) // 32

        prediction = self.model.predict(ds_test, steps=total)
        pr = np.argmax(prediction, axis=1)
        return np.reshape(pr, (X.shape[0] - self.windowSize, X.shape[1] - self.windowSize))
    # ------------------------------------------------------------------------------------------------------------------
