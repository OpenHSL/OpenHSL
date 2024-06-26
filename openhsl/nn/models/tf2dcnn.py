import numpy as np
import os
import tensorflow as tf
import wandb

from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from typing import Any, Dict, Optional, Union

from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask
from openhsl.nn.data.tf_dataloader import get_test_generator, get_train_val_gens
from openhsl.nn.data.utils import get_dataset
from openhsl.nn.models.utils import init_tensorboard, init_wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TF2DCNN:

    def __init__(self,
                 n_classes: int,
                 n_bands: int,
                 path_to_weights: str = None,
                 device: str = 'cpu'):

        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 5
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'tf2d'
        self.wandb_run = None
        self.writer = None

        self.train_loss = []
        self.val_loss = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = None

        input_shape = (self.hyperparams['n_bands'], self.hyperparams['patch_size'], self.hyperparams['patch_size'])

        C1 = 3 * self.hyperparams['n_bands']

        self.model = Sequential()

        self.model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(6 * self.hyperparams['n_bands'], activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.hyperparams['n_classes'], activation='softmax'))

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

    def init_wandb(self, path='../wandb.yaml'):
        self.wandb_run = init_wandb(path=path)
        if self.wandb_run:
            self.wandb_run.watch(self.model)
    # ------------------------------------------------------------------------------------------------------------------

    def init_tensorboard(self, path='tensorboard'):
        self.writer = init_tensorboard(path_dir=path)
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: Union[HSImage, np.ndarray],
            y: Union[HSMask, np.ndarray],
            fit_params: Dict):

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('batch_size', 32)
        # ToDo: add setdefault for optimizer, optimizer params and loss as in other models fit
        fit_params.setdefault('scheduler_type', None)
        fit_params.setdefault('scheduler_params', None)

        fit_params.setdefault('wandb', self.wandb_run)
        fit_params.setdefault('tensorboard', self.writer)

        fit_params.setdefault('wandb_vis', False)
        fit_params.setdefault('tensorboard_vis', False)

        img, gt = get_dataset(X, y)

        train_generator, val_generator = get_train_val_gens(X=img,
                                                            y=gt,
                                                            train_sample_percentage=fit_params['train_sample_percentage'],
                                                            patch_size=5)

        types = (tf.float32, tf.int32)
        shapes = ((self.hyperparams['n_bands'],
                   self.hyperparams['patch_size'],
                   self.hyperparams['patch_size']),
                  (self.hyperparams['n_classes'],))

        ds_train = tf.data.Dataset.from_generator(lambda: train_generator, types, shapes)
        ds_train = ds_train.batch(fit_params['batch_size']).repeat()
        ds_val = tf.data.Dataset.from_generator(lambda: val_generator, types, shapes)
        ds_val = ds_val.batch(fit_params['batch_size']).repeat()

        steps = len(train_generator) / fit_params['batch_size']
        val_steps = len(val_generator) / fit_params['batch_size']

        checkpoint_filepath = './checkpoints/tf2dcnn/'

        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)

        # add visualisations via callbacks
        callbacks = []

        if self.wandb_run:
            wandb_callback = wandb.keras.WandbCallback(monitor='val_loss',
                                                       log_evaluation=True)
            callbacks.append(wandb_callback)

        if fit_params['tensorboard_vis']:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard")
            callbacks.append(tensorboard_callback)

        history = self.model.fit(ds_train,
                                 validation_data=ds_val,
                                 validation_steps=val_steps,
                                 validation_batch_size=fit_params['batch_size'],
                                 batch_size=fit_params['batch_size'],
                                 epochs=fit_params['epochs'],
                                 steps_per_epoch=steps,
                                 verbose=1,
                                 callbacks=callbacks)

        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])
        self.train_accs = history.history.get('accuracy', [])
        self.val_accs = history.history.get('val_accuracy', [])

        self.model.save(f'{checkpoint_filepath}/weights.h5')

        if self.wandb_run:
            self.wandb_run.finish()
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: Union[HSImage, np.ndarray],
                y: Optional[HSMask] = None,
                batch_size=128) -> np.ndarray:

        types = tf.float32
        shapes = (self.hyperparams['n_bands'], self.hyperparams['patch_size'], self.hyperparams['patch_size'])

        img, gt = get_dataset(X, y)

        test_generator = get_test_generator(img, patch_size=self.hyperparams['patch_size'])
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(batch_size)

        total = sum([1 for _ in ds_test])

        test_generator = get_test_generator(img, patch_size=self.hyperparams['patch_size'])
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, types, shapes).batch(batch_size)

        prediction = self.model.predict(ds_test, steps=total)
        pr = np.argmax(prediction, axis=1)
        predicted_mask = np.reshape(pr, (X.data.shape[0], X.data.shape[1]))

        return predicted_mask
    # ------------------------------------------------------------------------------------------------------------------
