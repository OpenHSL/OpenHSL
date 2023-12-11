from abc import ABC, abstractmethod
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as udata
import numpy as np
import datetime
from tqdm import trange, tqdm

from openhsl.data.dataset import get_dataset
from openhsl.data.utils import camel_to_snake, grouper, count_sliding_window, \
                                        sliding_window, sample_gt, convert_to_color_
from openhsl.data.torch_dataloader import create_torch_loader
from openhsl.utils import init_wandb, init_tensorboard, EarlyStopping


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
        self.wandb_run = None
        self.writer = None

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

    def init_wandb(self, path='../wandb.yaml'):
        self.wandb_run = init_wandb(path=path)
        if self.wandb_run:
            self.wandb_run.watch(self.model)
    # ------------------------------------------------------------------------------------------------------------------

    def init_tensorboard(self, path='tensorboard'):
        self.writer = init_tensorboard(path_dir=path)
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

        if fit_params['scheduler_type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=fit_params['optimizer'],
                                                  **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=fit_params['optimizer'],
                                                             **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=fit_params['optimizer'],
                                                             **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] is None:
            scheduler = None
        else:
            raise ValueError('Unsupported scheduler type')

        fit_params.setdefault('wandb', None)
        fit_params.setdefault('tensorboard', None)

        train_gt, _ = sample_gt(gt=gt,
                                train_size=fit_params['train_sample_percentage'],
                                mode=fit_params['dataloader_mode'],
                                msg='train_val/test')

        train_gt, val_gt = sample_gt(gt=train_gt,
                                     train_size=0.9,
                                     mode=fit_params['dataloader_mode'],
                                     msg='train/val')

        print(f'Full size: {np.sum(gt > 0)}')
        print(f'Train size: {np.sum(train_gt > 0)}')
        print(f'Val size: {np.sum(val_gt > 0)}')

        # Generate the dataset

        train_loader = create_torch_loader(img, train_gt, hyperparams, shuffle=True)
        val_loader = create_torch_loader(img, val_gt, hyperparams)

        Model.save_train_mask(model_name=camel_to_snake(str(model.__class__.__name__)),
                              dataset_name=train_loader.dataset.name,
                              mask=train_gt)

        model, history = Model.train(net=model,
                                     optimizer=fit_params['optimizer'],
                                     criterion=fit_params['loss'],
                                     scheduler=scheduler,
                                     data_loader=train_loader,
                                     epoch=fit_params['epochs'],
                                     val_loader=val_loader,
                                     wandb_run=fit_params['wandb'],
                                     writer=fit_params['tensorboard'],
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
              scheduler: torch.optim.lr_scheduler,
              data_loader: udata.DataLoader,
              epoch,
              wandb_run=None,
              writer=None,
              device=None,
              val_loader=None,
              ):
        """
        Training loop to optimize a network for several epochs and a specified loss
        Parameters
        ----------
            net:
                a PyTorch model
            optimizer:
                a PyTorch optimizer
            criterion:
                a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
            scheduler:
                PyTorch scheduler
            data_loader:
                a PyTorch dataset loader
            epoch:
                int specifying the number of training epochs
            display_iter:
                number of iterations before refreshing the display (False/None to switch off).
            device:
                torch device to use (defaults to CPU)
            val_loader:
                validation dataset
            wandb_vis:
                Flag to enable weights & biases visualisation
            tensorboard_vis:
                Flag to enable Tensorboard logging and visualisation
            wandb_path:
                ...
            tensorboard_path:
                ...
        """
        net.to(device)

        #if wandb_vis:
        #    wandb_run = init_wandb(path=wandb_path)
        #    if wandb_run:
        #        wandb_run.watch(net)

        #if tensorboard_vis:
        #    writer = init_tensorboard(path_dir=tensorboard_path)

        save_epoch = epoch // 20 if epoch > 20 else 1

        early_stopping = EarlyStopping(tolerance=5, min_delta=10)

        losses = []
        train_accuracies = []
        val_accuracies = []
        train_loss = []
        val_loss = []
        t = trange(1, epoch + 1, desc='Train loop', leave=True, dynamic_ncols=True)
        for e in t:
            # Set the network to training mode
            net.train()
            avg_train_loss = 0.0
            train_accuracy = 0.0
            total = 0
            # Run the training loop for one epoch
            for batch_idx, (data, target) in (enumerate(data_loader)):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                avg_train_loss += loss.item()
                losses.append(loss.item())
                _, output = torch.max(output, dim=1)
                for out, pred in zip(output.view(-1), target.view(-1)):
                    if out.item() in [0]:
                        continue
                    else:
                        train_accuracy += out.item() == pred.item()
                        total += 1

            # Update the scheduler
            avg_train_loss /= len(data_loader)
            train_loss.append(avg_train_loss)
            train_acc = train_accuracy / total
            train_accuracies.append(train_acc)

            if val_loader: # ToDo: not preferable to check if condition every iteration
                val_acc, avg_val_loss = Model.val(net, criterion, val_loader, device=device)

                t.set_postfix({'train_acc': "{:.3f}".format(train_acc),
                               'val_acc': "{:.3f}".format(val_acc),
                               'train_loss': "{:.3f}".format(avg_train_loss),
                               'val_loss': "{:.3f}".format(avg_val_loss)
                               }
                              )

                val_loss.append(avg_val_loss)
                val_accuracies.append(val_acc)
                metric = -val_acc  # ToDo WTF
            else:
                metric = avg_train_loss

            # Update the scheduler (ToDo: not preferable to check if condition every iteration)
            if scheduler is not None:
                scheduler.step()

            # log metrics
            if wandb_run:
                wandb_run.log({"Loss/train": avg_train_loss,
                                   "Loss/val": avg_val_loss,
                                   "Accuracy/train": train_acc,
                                   "Accuracy/val": val_acc,
                                   "Learning rate": optimizer.param_groups[0]['lr']
                                }
                             )

            if writer:
                net.writer.add_scalar('Loss/train', avg_train_loss, e)
                net.writer.add_scalar('Loss/val', avg_val_loss, e)
                net.writer.add_scalar('Accuracy/train', train_acc, e)
                net.writer.add_scalar('Accuracy/val', val_acc, e)
                net.writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], e)

            # Save the weights
            if e % save_epoch == 0:
                Model.save_model(
                    net,
                    camel_to_snake(str(net.__class__.__name__)),
                    data_loader.dataset.name,
                    epoch=e,
                    metric=abs(metric),
                )

            # Early stopping
            early_stopping(avg_train_loss, avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if wandb_run:
            wandb_run.finish()

        if writer:
            writer.close()

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

        # ToDo: Refactor def test(), merge with val if possible

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
                data = data.to(device, dtype=torch.float)
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
        gray_filename = f"{mask_dir}/{time_str}_gray_mask.npy"
        color_filename = f"{mask_dir}/{time_str}_color_mask.png"
        #color = Image.fromarray(convert_to_color_(mask))
        np.save(gray_filename, mask)
        #color.save(color_filename)
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
