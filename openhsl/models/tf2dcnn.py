import os
import copy
import numpy as np

from typing import Optional, Dict

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.optimizers import SGD

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.data.utils import apply_pca
from openhsl.data.tf_dataloader import preprocess_data, get_data_generator, get_test_generator


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TF2DCNN:

    def __init__(self,
                 n_classes: int,
                 n_bands: int,
                 apply_pca=False,
                 path_to_weights: str = None,
                 device: str = 'cpu'):

        self.train_loss = []
        self.val_loss = []
        self.train_accs = []
        self.val_accs = []

        self.patch_size = 5
        self.n_bands = n_bands
        self.class_count = n_classes - 1
        self.apply_pca = apply_pca
        input_shape = (self.n_bands, self.patch_size, self.patch_size)

        C1 = 3 * self.n_bands

        self.model = Sequential()

        self.model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(6 * self.n_bands, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.class_count, activation='softmax'))

        sgd = SGD(learning_rate=0.0001,
                  decay=1e-6,
                  momentum=0.9,
                  nesterov=True)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        if path_to_weights:
            self.model.load_weights(path_to_weights)
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: HSImage,
            y: HSMask,
            fit_params: Dict):

        if self.apply_pca:
            X = copy.deepcopy(X)
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

        history = self.model.fit(ds_train,
                                 validation_data=ds_val,
                                 validation_steps=val_steps,
                                 validation_batch_size=fit_params['batch_size'],
                                 batch_size=fit_params['batch_size'],
                                 epochs=fit_params['epochs'],
                                 steps_per_epoch=steps,
                                 verbose=1)

        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])
        self.train_accs = history.history.get('accuracy', [])
        self.val_accs = history.history.get('val_accuracy', [])

        self.model.save(f'{checkpoint_filepath}/weights.h5')
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: HSImage,
                y: Optional[HSMask] = None) -> np.ndarray:

        if self.apply_pca:
            X = copy.deepcopy(X)
            X.data, _ = apply_pca(X.data, self.n_bands)
        else:
            print('PCA will not apply')

        types = tf.float32
        shapes = (self.n_bands, self.patch_size, self.patch_size)
        X = X.data

        test_generator = get_test_generator(X, patch_size=self.patch_size)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(128)

        # TODO bad issue
        total = sum([1 for i in ds_test])

        test_generator = get_test_generator(X, patch_size=self.patch_size)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(128)

        prediction = self.model.predict(ds_test, steps=total)
        pr = np.argmax(prediction, axis=1)
        predicted_mask = np.reshape(pr, (X.shape[0], X.shape[1]))

        return predicted_mask + 1
    # ------------------------------------------------------------------------------------------------------------------
