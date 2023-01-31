import model

from Firsov_Legacy.DataLoader import DataLoader
from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.Models.model import get_model
from Firsov_Legacy.model_tvts import train, test
from Firsov_Legacy.utils import sample_gt, convert_to_color_

import numpy as np
import torch
import torch.utils.data as data


def create_loader(img: np.array,
                  gt: np.array,
                  hyperparams: dict,
                  shuffle: bool = False):
    dataset = DataLoader(img, gt, **hyperparams)
    return data.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)


class M3DCNN:

    def __init__(self):
        pass

    def fit(self,
            X,
            y,
            hyperparams: dict,
            epochs: int = 5,
            sample_percentage: float = 0.5,
            weights_path=None):

        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(hsi=X, mask=y)

        hyperparams['patch_size'] = 7
        hyperparams['batch_size'] = 40
        hyperparams['learning_rate'] = 0.01
        hyperparams['n_bands'] = img.shape[-1]
        hyperparams['ignored_labels'] = IGNORED_LABELS
        hyperparams['net_name'] = 'he'
        hyperparams['n_classes'] = len(np.unique(gt))
        hyperparams['epoch'] = epochs

        model, optimizer, loss, hyperparams = get_model(hyperparams)

        if weights_path:
            model.load_state_dict(torch.load(weights_path))

        train_gt, _ = sample_gt(gt, sample_percentage, mode='random')

        train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")

        # Generate the dataset
        train_loader = create_loader(img, train_gt, hyperparams, shuffle=True)

        val_loader = create_loader(img, val_gt, hyperparams)

        train(
            model,
            optimizer,
            loss,
            train_loader,
            hyperparams["epoch"],
            scheduler=hyperparams["scheduler"],
            device=hyperparams["device"],
            val_loader=val_loader,
        )

    def predict(self,
                X,
                hyperparams: dict,
                weights_path: str,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(X, mask=None)

        hyperparams['patch_size'] = 7
        hyperparams['batch_size'] = 40
        hyperparams['n_classes'] = len(LABEL_VALUES)
        hyperparams['n_bands'] = img.shape[-1]
        hyperparams['ignored_labels'] = IGNORED_LABELS
        hyperparams['learning_rate'] = 0.01
        hyperparams['test_stride'] = 1

        model, optimizer, loss, hyperparams = get_model(hyperparams)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        color_prediction = convert_to_color_(prediction, palette)

        return gt, prediction, color_prediction
