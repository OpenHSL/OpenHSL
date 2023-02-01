import model
import os
from Firsov_Legacy.DataLoader import DataLoader
from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.Models.model import get_model
from Firsov_Legacy.utils import sample_gt, convert_to_color_, get_device, \
                                camel_to_snake, grouper, count_sliding_window, sliding_window

import numpy as np
import torch
import torch.utils.data as data
import torch.optim
from tqdm import tqdm
import datetime

from hsi import HSImage
from hs_mask import HSMask

def create_loader(img: np.array,
                  gt: np.array,
                  hyperparams: dict,
                  shuffle: bool = False):
    dataset = DataLoader(img, gt, **hyperparams)
    return data.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)


class M3DCNN:

    def __init__(self,
                 n_classes=3,
                 n_bands=250,
                 patch_size=7,
                 path_to_weights=None,
                 device='cpu'
                 ):
        self.hyperparams = {}
        self.hyperparams['patch_size'] = patch_size
        self.hyperparams['batch_size'] = 40
        self.hyperparams['learning_rate'] = 0.01
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'he'
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device

        self.model, self.optimizer, self.loss, self.hyperparams = get_model(self.hyperparams)
        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

    def fit(self,
            X: HSImage,
            y: HSMask,
            epochs: int = 5,
            sample_percentage: float = 0.5):

        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(hsi=X, mask=y)

        self.hyperparams['epoch'] = epochs

        train_gt, _ = sample_gt(gt, sample_percentage, mode='random')
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")

        # Generate the dataset
        train_loader = create_loader(img, train_gt, self.hyperparams, shuffle=True)
        val_loader = create_loader(img, val_gt, self.hyperparams)

        self.train(data_loader=train_loader,
                   epoch=epochs,
                   val_loader=val_loader,
                   device=self.hyperparams['device']
        )

    def predict(self,
                X: HSImage,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.hyperparams["test_stride"] = 1
        img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(X, mask=None)

        self.model.eval()

        probabilities = self.test(img, self.hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        color_prediction = convert_to_color_(prediction, palette)

        return gt, prediction, color_prediction

    def train(self,
              data_loader: data.DataLoader,
              epoch,
              display_iter=100,
              device=torch.device("cpu"),
              val_loader=None
    ):
        """
        Training loop to optimize a network for several epochs and a specified loss

        Parameters
        ----------
            optimizer: a PyTorch optimizer
            data_loader: a PyTorch dataset loader
            epoch: int specifying the number of training epochs
            criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            device (optional): torch device to use (defaults to CPU)
            display_iter (optional): number of iterations before refreshing the
            display (False/None to switch off).
            scheduler (optional): PyTorch scheduler
            val_loader (optional): validation dataset
            supervision (optional): 'full' or 'semi'
        """
        self.model.to(device)
        save_epoch = epoch // 20 if epoch > 20 else 1
        losses = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        iter_ = 1
        val_accuracies = []

        for e in tqdm(range(1, epoch + 1)):
            self.model.train()
            avg_loss = 0.0
            for batch_idx, (data, target) in (enumerate(data_loader)):
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()

                output = self.model(data)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()

                avg_loss += loss.item()
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100): iter_ + 1])

                if display_iter and iter_ % display_iter == 0:
                    string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                    string = string.format(
                        e,
                        epoch,
                        batch_idx * len(data),
                        len(data) * len(data_loader),
                        100.0 * batch_idx / len(data_loader),
                        mean_losses[iter_],
                    )

                    tqdm.write(string)

                iter_ += 1
                del (data, target, loss, output)

            # Update the scheduler
            avg_loss /= len(data_loader)
            if val_loader is not None:
                val_acc = self.val(val_loader, device=device)
                val_accuracies.append(val_acc)
                metric = -val_acc
            else:
                metric = avg_loss

            # Save the weights
            if e % save_epoch == 0:
                self.save_model(
                    camel_to_snake(str(self.model.__class__.__name__)),
                    data_loader.dataset.name,
                    epoch=e,
                    metric=abs(metric),
                )

    def val(self,
            data_loader: data.DataLoader,
            device: torch.device):
        # TODO : fix me using metrics()
        accuracy, total = 0.0, 0.0
        ignored_labels = self.hyperparams['ignored_labels']
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total

    def test(self,
             img: np.array,
             hyperparams: dict):
        """
        Test a model on a specific image
        """
        """
            Test a model on a specific image
            """
        patch_size = hyperparams["patch_size"]
        center_pixel = hyperparams["center_pixel"]
        batch_size, device = hyperparams["batch_size"], hyperparams["device"]
        n_classes = hyperparams["n_classes"]

        kwargs = {
            "step": hyperparams["test_stride"],
            "window_size": (patch_size, patch_size),
        }
        probs = np.zeros(img.shape[:2] + (n_classes,))

        iterations = count_sliding_window(img, **kwargs) // batch_size
        for batch in tqdm(
                grouper(batch_size, sliding_window(img, **kwargs)),
                total=iterations,
                desc="Inference on the image",
        ):
            with torch.no_grad():
                if patch_size == 1:
                    data = [b[0][0, 0] for b in batch]
                    data = np.copy(data)
                    data = torch.from_numpy(data)
                else:
                    data = [b[0] for b in batch]
                    data = np.copy(data)
                    data = data.transpose(0, 3, 1, 2)
                    data = torch.from_numpy(data)
                    data = data.unsqueeze(1)

                indices = [b[1:] for b in batch]
                data = data.to(device)
                output = self.model(data)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.to("cpu")

                if patch_size == 1 or center_pixel:
                    output = output.numpy()
                else:
                    output = np.transpose(output.numpy(), (0, 2, 3, 1))
                for (x, y, w, h), out in zip(indices, output):
                    if center_pixel:
                        probs[x + w // 2, y + h // 2] += out
                    else:
                        probs[x: x + w, y: y + h] += out
        return probs

    def save_model(self,
                   model_name,
                   dataset_name,
                   **kwargs):
        model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
        """
        Using strftime in case it triggers exceptions on windows 10 system
        """
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        if isinstance(self.model, torch.nn.Module):
            filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
                **kwargs
            )
            tqdm.write("Saving neural network weights in {}".format(filename))
            torch.save(self.model.state_dict(), model_dir + filename + ".pth")
        else:
            print('Saving error')
