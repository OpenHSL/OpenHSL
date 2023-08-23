from abc import ABC, abstractmethod
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as udata
import numpy as np
import datetime
from tqdm import trange, tqdm
from PIL import Image

from openhsl.data.dataset import get_dataset
from openhsl.data.utils import camel_to_snake, grouper, count_sliding_window, \
                                        sliding_window, sample_gt, convert_to_color_
from openhsl.data.torch_dataloader import create_loader


class Model(ABC):
    """
    Model()

        Abstract class for decorating machine learning algorithms

    """

    @abstractmethod
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_accs = []
        self.val_accs = []
        self.model = None

    @abstractmethod
    def fit(self,
            X,
            y,
            fit_params):
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
               fit_params):
        """

        Parameters
        ----------
        X
        y
        hyperparams
        model
        fit_params

        Returns
        -------

        """
        # TODO ignored_labels and label_values for what?
        img, gt = get_dataset(hsi=X, mask=y)

        hyperparams['batch_size'] = fit_params['batch_size']

        train_gt, _ = sample_gt(gt=gt,
                                train_size=fit_params['train_sample_percentage'],
                                mode=fit_params['dataloader_mode'])

        train_gt, val_gt = sample_gt(gt=train_gt,
                                     train_size=0.9,
                                     mode=fit_params['dataloader_mode'])

        print(f'Full size: {np.sum(gt > 0)}')
        print(f'Train size: {np.sum(train_gt > 0)}')
        print(f'Val size: {np.sum(val_gt > 0)}')

        # Generate the dataset

        train_loader = create_loader(img, train_gt, hyperparams, shuffle=True)
        val_loader = create_loader(img, val_gt, hyperparams)

        Model.save_train_mask(model_name=camel_to_snake(str(model.__class__.__name__)),
                              dataset_name=train_loader.dataset.name,
                              mask=train_gt)

        model, history = Model.train(net=model,
                                     optimizer=fit_params['optimizer'],
                                     criterion=fit_params['loss'],
                                     data_loader=train_loader,
                                     epoch=fit_params['epochs'],
                                     val_loader=val_loader,
                                     device=hyperparams['device'])
        return model, history
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def predict_nn(model,
                   X,
                   y=None,
                   hyperparams=None):
        hyperparams["test_stride"] = 1
        hyperparams.setdefault('batch_size', 1)
        img, gt = get_dataset(X, mask=None)

        model.eval()

        probabilities = Model.test(net=model,
                                   img=img,
                                   hyperparams=hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        # fill void areas in result with zeros
        if y:
            prediction[y.get_2d() == 0] = 0
        return prediction
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def train(net: nn.Module,
              optimizer: torch.optim,
              criterion,
              data_loader: udata.DataLoader,
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

        losses = []
        train_accuracies = []
        val_accuracies = []
        train_loss = []
        val_loss = []
        t = trange(1, epoch + 1, desc='Train loop', leave=True)
        for e in t:
            # Set the network to training mode
            net.train()
            avg_loss = 0.0
            accuracy = 0.0
            total = 0
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
                losses.append(loss.item())
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in [0]:
                        continue
                    else:
                        accuracy += out.item() == pred.item()
                        total += 1

            # Update the scheduler
            avg_loss /= len(data_loader)
            train_loss.append(avg_loss)
            train_acc = accuracy / total
            train_accuracies.append(train_acc)

            if val_loader:
                val_acc, loss = Model.val(net, criterion, val_loader, device=device)

                t.set_postfix_str(f"train accuracy: {train_acc}\t"
                                  f"val accuracy: {val_acc}\t"
                                  f"train loss: {avg_loss}\t"
                                  f"val loss: {loss}")
                t.refresh()

                val_loss.append(loss)
                val_accuracies.append(val_acc)
                metric = -val_acc  # TODO WTF
            else:
                metric = avg_loss

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            elif scheduler:
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
        history = dict()
        history["train_loss"] = train_loss
        history["val_loss"] = val_loss
        history["train_accuracy"] = train_accuracies
        history["val_accuracy"] = val_accuracies
        return net, history
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def val(net: nn.Module,
            criterion,
            data_loader: udata.DataLoader,
            device: torch.device):
        # TODO : fix me using metrics()
        val_accuracy, total = 0.0, 0.0
        avg_loss = 0
        ignored_labels = data_loader.dataset.ignored_labels
        for batch_idx, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                # Load the data into the GPU if required
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target)
                avg_loss += loss.item()
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in ignored_labels:
                        continue
                    else:
                        val_accuracy += out.item() == pred.item()
                        total += 1

        return val_accuracy / total, avg_loss / len(data_loader)
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
    def save_train_mask(model_name, dataset_name, mask):

        mask_dir = "./masks/" + model_name + "/" + dataset_name + "/"
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.isdir(mask_dir):
            os.makedirs(mask_dir, exist_ok=True)
        gray_filename = f"{mask_dir}/{time_str}_gray_mask.png"
        color_filename = f"{mask_dir}/{time_str}_color_mask.png"
        gray = Image.fromarray(mask)
        color = Image.fromarray(convert_to_color_(mask))
        gray.save(gray_filename)
        color.save(color_filename)
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
            torch.save(model.state_dict(), model_dir + filename + ".pth")
        else:
            print('Saving error')
    # ------------------------------------------------------------------------------------------------------------------
