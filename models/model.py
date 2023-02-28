from abc import ABC, abstractmethod
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import datetime
from tqdm import tqdm

from Firsov_Legacy.dataset import get_dataset
from Firsov_Legacy.utils import camel_to_snake, grouper, count_sliding_window, sliding_window, sample_gt
from Firsov_Legacy.DataLoader import create_loader


class Model(ABC):
    """
    Model()

        Abstract class for decorating machine learning algorithms

    """

    @abstractmethod
    def fit(self,
            X,
            y,
            epochs):
        raise NotImplemented("Method fit must be implemented!")
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def predict(self,
                X,
                y) -> np.ndarray:
        raise NotImplemented("Method predict must be implemented!")
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def fit_nn(X,
               y,
               hyperparams,
               model,
               optimizer,
               loss,
               epochs,
               train_sample_percentage):
        # TODO ignored_labels and label_values for what?
        img, gt = get_dataset(hsi=X, mask=y)

        hyperparams['epoch'] = epochs

        train_gt, _ = sample_gt(gt, train_sample_percentage, mode='random')
        train_gt, val_gt = sample_gt(train_gt, 0.9, mode="random")

        print(f'Full size: {np.sum(gt > 0)}')
        print(f'Train size: {np.sum(train_gt > 0)}')
        print(f'Val size: {np.sum(val_gt > 0)}')

        # Generate the dataset

        train_loader = create_loader(img, train_gt, hyperparams, shuffle=True)
        val_loader = create_loader(img, val_gt, hyperparams)

        model, losses = Model.train(net=model,
                            optimizer=optimizer,
                            criterion=loss,
                            data_loader=train_loader,
                            epoch=epochs,
                            val_loader=val_loader,
                            device=hyperparams['device'])
        return model, losses
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def predict_nn(X,
                   y,
                   hyperparams,
                   model):
        hyperparams["test_stride"] = 1
        img, gt = get_dataset(X, mask=None)

        model.eval()

        probabilities = Model.test(net=model,
                                   img=img,
                                   hyperparams=hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        # fill void areas in result with zeros
        if y:
            prediction[y.data == 0] = 0
        return prediction
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def train(net: nn.Module,
              optimizer: torch.optim,
              criterion,
              data_loader: data.DataLoader,
              epoch,
              scheduler=None,
              display_iter=100,
              device=None,
              val_loader=None):
        """
        Training loop to optimize a network for several epochs and a specified loss
        Parameters
        ----------
            net:
                a PyTorch model
            optimizer:
                a PyTorch optimizer
            data_loader:
                a PyTorch dataset loader
            epoch:
                int specifying the number of training epochs
            criterion:
                a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            device (optional):
                torch device to use (defaults to CPU)
            display_iter (optional):
                number of iterations before refreshing the display (False/None to switch off).
            scheduler (optional):
                PyTorch scheduler
            val_loader (optional):
                validation dataset
            supervision (optional):
                'full' or 'semi'
        """
        net.to(device)

        save_epoch = epoch // 20 if epoch > 20 else 1

        losses = np.zeros(1000000)
        mean_losses = np.zeros(100000000)
        iter_ = 1
        val_accuracies = []
        train_loss = []
        for e in tqdm(range(1, epoch + 1)):
            # Set the network to training mode
            net.train()
            avg_loss = 0.0

            # Run the training loop for one epoch
            for batch_idx, (data, target) in (enumerate(data_loader)):
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
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
                    # TODO remake it
                    # tqdm.write(string)

                iter_ += 1
                del (data, target, loss, output)

            # Update the scheduler
            avg_loss /= len(data_loader)
            train_loss.append(avg_loss)
            if val_loader is not None:
                val_acc = Model.val(net, val_loader, device=device)
                tqdm.write(f"val accuracy: {val_acc}")
                val_accuracies.append(val_acc)
                metric = -val_acc
            else:
                metric = avg_loss

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            elif scheduler is not None:
                scheduler.step()

            # Save the weights
            if e % save_epoch == 0:
                Model.save_model(
                    net,
                    camel_to_snake(str(net.__class__.__name__)),
                    data_loader.dataset.name,
                    epoch=e,
                    metric=abs(metric),
                )
        return net, train_loss
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def val(net: nn.Module,
            data_loader: data.DataLoader,
            device: torch.device):
        # TODO : fix me using metrics()
        accuracy, total = 0.0, 0.0
        ignored_labels = data_loader.dataset.ignored_labels
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                output = net(data)
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1
        return accuracy / total
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def test(net: nn.Module,
             img: np.array,
             hyperparams):
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
        net.to(device)
        iterations = count_sliding_window(img, **kwargs) // batch_size
        for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                          total=iterations,
                          desc="Inference on the image"):
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
                output = net(data)
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
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def save_model(model,
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
        if isinstance(model, torch.nn.Module):
            filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
                **kwargs
            )
            # TODO remake it
            # tqdm.write("Saving neural network weights in {}".format(filename))
            torch.save(model.state_dict(), model_dir + filename + ".pth")
        else:
            print('Saving error')
    # ------------------------------------------------------------------------------------------------------------------
