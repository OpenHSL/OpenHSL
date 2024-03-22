import os
import sys
import torch
import torch.nn as nn
import tensorflow as tf
from keras.callbacks import Callback
from tqdm import trange
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QFileDialog)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot

from gui.utils import (get_file_name, get_file_extension, get_date_time, create_dir_if_not_exist,
                       request_keys_from_h5_file, request_keys_from_mat_file,
                       get_gpu_info)
from gui.common_gui import CIU
from gui.mac_trainer_gui import Ui_MainWindow

from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask
from openhsl.data.dataset import get_dataset
from openhsl.data.torch_dataloader import create_torch_loader
from openhsl.models.model import train_one_epoch, val_one_epoch, get_optimizer, get_scheduler
from openhsl.data.utils import get_palette, convert_to_color_, sample_gt

from openhsl.models.ssftt import SSFTT
from openhsl.models.m1dcnn import M1DCNN
from openhsl.models.m3dcnn_li import M3DCNN as LI
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.data.tf_dataloader import get_test_generator, get_train_val_gens
from openhsl.models.tf2dcnn import TF2DCNN


models_dict = {
    "M1DCNN": M1DCNN,
    "M3DCNN_li": LI,
    "NM3DCNN": NM3DCNN,
    "SSFTT": SSFTT,
    "TF2DCNN": TF2DCNN
}

stop_train = False


tf_train_params = {}


class SendStatsCallback(Callback):

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def on_epoch_end(self, epoch, logs=None):
        global tf_train_params
        tf_train_params['loss'] = logs['loss']
        tf_train_params['val_loss'] = logs['val_loss']

        tf_train_params['accuracy'] = logs['accuracy']
        tf_train_params['val_accuracy'] = logs['val_accuracy']

        self.signal.emit({"epoch": epoch + 1,
                          "val_loss": logs['val_loss'],
                          "train_loss": logs['loss'],
                          "val_acc": logs['val_accuracy'],
                          "train_acc": logs['accuracy'],
                          "model": "TF2DCNN",
                          "run_name": "TF2DCNN"})


class InferenceWorker(QObject):
    progress_signal = Signal(dict)

    @Slot(dict)
    def do_inference(self, fits):
        try:
            hsi = fits["hsi"]
            path_weights = fits["weights"]
            device = fits["device"]

            weights = torch.load(path_weights, map_location=device)
            num_classes = len(weights[next(reversed(weights))])
            del weights

            net = fits["model"](n_classes=num_classes,
                                device=device,
                                n_bands=hsi.data.shape[-1],
                                path_to_weights=path_weights)

            predict = net.predict(hsi)
            self.progress_signal.emit({"predict": predict})

        except Exception as e:
            self.progress_signal.emit({"error": str(e)})


class TrainWorker(QObject):
    progress_signal = Signal(dict)

    @Slot(dict)
    def do_train(self, fits):
        global stop_train
        try:
            if fits["model"].__name__ != 'TF2DCNN':
                net = fits["model"](n_bands=fits["hsi"].data.shape[-1],
                                    n_classes=fits["mask"].n_classes,
                                    path_to_weights=fits["weights"],
                                    device=fits["device"])

                net.hyperparams["batch_size"] = fits["fit_params"]["batch_size"]
                img, gt = get_dataset(fits["hsi"], fits["mask"])
                train_gt, _ = sample_gt(gt=gt,
                                        train_size=fits["fit_params"]['train_sample_percentage'],
                                        mode=fits["fit_params"]['dataloader_mode'],
                                        msg='train_val/test')

                train_gt, val_gt = sample_gt(gt=train_gt,
                                             train_size=0.9,
                                             mode=fits["fit_params"]['dataloader_mode'],
                                             msg='train/val')

                train_data_loader = create_torch_loader(img,
                                                        train_gt,
                                                        net.hyperparams,
                                                        shuffle=True)

                val_data_loader = create_torch_loader(img,
                                                      val_gt,
                                                      net.hyperparams)

                criterion = nn.CrossEntropyLoss(weight=net.hyperparams.get("weights", None))
                optimizer = get_optimizer(net=net.model,
                                          optimizer_type='SGD',
                                          optimizer_params=fits["optimizer_params"])

                scheduler = get_scheduler(scheduler_type=fits["fit_params"]['scheduler_type'],
                                          optimizer=optimizer,
                                          scheduler_params=fits["scheduler_params"])

                device = net.hyperparams['device']
                net.model.to(device)
                for e in trange(fits["fit_params"]['epochs']):
                    temp_train = train_one_epoch(net=net.model,
                                                 criterion=criterion,
                                                 data_loader=train_data_loader,
                                                 optimizer=optimizer,
                                                 device=device)
                    if scheduler is not None:
                        scheduler.step()

                    temp_val = val_one_epoch(net=net.model,
                                             criterion=criterion,
                                             data_loader=val_data_loader,
                                             device=device)

                    self.progress_signal.emit({"epoch": e + 1,
                                               "val_loss": temp_val[1],
                                               "train_loss": temp_train['avg_train_loss'],
                                               "val_acc": temp_val[0],
                                               "train_acc": temp_train['train_acc'],
                                               "model": net,
                                               "run_name": fits["run_name"]})

                    if stop_train:
                        stop_train = False
                        break

                self.progress_signal.emit({"end": True,
                                           "model": net,
                                           "run_name": fits["run_name"]})

            else:
                net = fits["model"](n_bands=fits["hsi"].data.shape[-1],
                                    n_classes=fits["mask"].n_classes,
                                    path_to_weights=fits["weights"],
                                    device=fits["device"])
                img, gt = get_dataset(fits["hsi"], fits["mask"])

                train_generator, val_generator = get_train_val_gens(X=img,
                                                                    y=gt,
                                                                    train_sample_percentage=fits["fit_params"]['train_sample_percentage'],
                                                                    patch_size=5)

                types = (tf.float32, tf.int32)
                shapes = ((net.hyperparams['n_bands'],
                           net.hyperparams['patch_size'],
                           net.hyperparams['patch_size']),
                          (net.hyperparams['n_classes'],))

                ds_train = tf.data.Dataset.from_generator(lambda: train_generator, types, shapes)
                ds_train = ds_train.batch(fits["fit_params"]["batch_size"]).repeat()
                ds_val = tf.data.Dataset.from_generator(lambda: val_generator, types, shapes)
                ds_val = ds_val.batch(fits["fit_params"]["batch_size"]).repeat()

                steps = len(train_generator) / fits["fit_params"]["batch_size"]
                val_steps = len(val_generator) / fits["fit_params"]["batch_size"]

                checkpoint_filepath = './checkpoints/tf2dcnn/'

                if not os.path.exists(checkpoint_filepath):
                    os.makedirs(checkpoint_filepath)

                # add visualisations via callbacks
                callbacks = []

                callbacks.append(SendStatsCallback(self.progress_signal))

                history = net.model.fit(ds_train,
                                        validation_data=ds_val,
                                        validation_steps=val_steps,
                                        validation_batch_size=fits["fit_params"]['batch_size'],
                                        batch_size=fits["fit_params"]['batch_size'],
                                        epochs=fits["fit_params"]['epochs'],
                                        steps_per_epoch=steps,
                                        verbose=1,
                                        callbacks=callbacks)

                #self.train_loss = history.history.get('loss', [])
                #self.val_loss = history.history.get('val_loss', [])
                #self.train_accs = history.history.get('accuracy', [])
                #self.val_accs = history.history.get('val_accuracy', [])

                net.model.save(f'{checkpoint_filepath}/weights.h5')

        except Exception as e:
            self.progress_signal.emit({"error": str(e)})


class MainWindow(CIU):
    progress_requested = Signal(dict)
    inference_requested = Signal(dict)

    def __init__(self):
        CIU.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.current_data = {}
        self.current_hsi = None
        self.current_mask = None
        self.current_predict = None
        self.current_key = None
        self.imported_weights = {}
        self.show()

        devices = get_gpu_info()
        self.devices_dict = {"cpu": "cpu"}
        devices.remove("cpu")

        for i, device in enumerate(devices):
            self.devices_dict[device] = i

        devices.append("cpu")
        self.ui.device_box.addItems(devices)
        self.ui.device_box2.addItems(devices)
        self.current_device = self.ui.device_box.currentText()

        self.current_train_hsi = None
        self.current_train_mask = None

        self.current_test_hsi = None
        self.current_test_mask = None
        self.current_test_predict = None
        self.current_test_key = None
        self.current_classification_report = None
        self.current_confusion_matrix = None

        # THREAD INFERENCE SETUP
        self.inference_worker = InferenceWorker()
        self.inference_thread = QThread()

        self.inference_worker.progress_signal.connect(self.update_inference_progress)
        self.inference_requested.connect(self.inference_worker.do_inference)

        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.start()

        # THREAD TRAIN SETUP
        self.train_worker = TrainWorker()
        self.train_thread = QThread()

        self.train_worker.progress_signal.connect(self.update_train_progress)
        self.progress_requested.connect(self.train_worker.do_train)

        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.start()

        # NAVIGATION
        self.ui.trainer_mod_btn.clicked.connect(lambda:
                                                self.ui.stackedWidget.setCurrentWidget(self.ui.trainer_widget))

        self.ui.inference_mod_btn.clicked.connect(lambda:
                                                  self.ui.stackedWidget.setCurrentWidget(self.ui.inference_widget))

        self.ui.show_loss_btn.clicked.connect(lambda:
                                              self.ui.stacked_graphs.setCurrentWidget(self.ui.loss_page))

        self.ui.show_accuracy_btn.clicked.connect(lambda:
                                                  self.ui.stacked_graphs.setCurrentWidget(self.ui.acc_page))

        # BUTTON CONNECTIONS
        self.ui.import_data_btn.clicked.connect(self.extract_data)
        self.ui.add_mask_btn.clicked.connect(self.add_mask)
        self.ui.Show.clicked.connect(self.show_image)
        self.ui.data_to_train_btn.clicked.connect(self.data_to_train)
        self.ui.data_to_test_btn.clicked.connect(self.data_to_test)
        self.ui.start_learning_btn.clicked.connect(self.start_learning)
        self.ui.browse_weight_for_inference_btn.clicked.connect(self.import_weights)
        self.ui.stop_train_btn.clicked.connect(self.stop_train)
        self.ui.start_inference_btn.clicked.connect(self.start_inference)
        self.ui.estimate_btn.clicked.connect(self.estimate)

        # OTHER INTERACT
        self.ui.horizontalSlider.valueChanged.connect(
            lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                              self.ui.highcontast_check.isChecked(),
                                              self.ui.spin_channels.value(),
                                              self.ui.image_label))
        self.ui.spin_channels.valueChanged.connect(
            lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                              self.ui.highcontast_check.isChecked(),
                                              self.ui.spin_channels.value(),
                                              self.ui.image_label))
        self.ui.highcontast_check.stateChanged.connect(
            lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                              self.ui.highcontast_check.isChecked(),
                                              self.ui.spin_channels.value(),
                                              self.ui.image_label))

        self.ui.image_view_box.currentIndexChanged.connect(self.view_changed)
        self.ui.cm_slider.valueChanged.connect(self.update_cm)

    def import_weights(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                    "Select a weight file",
                                                    "",
                                                    "(*.pth);;(*.pt);;(*.h5)",
                                                    options=QFileDialog.Options())
        if file_name:
            name = get_file_name(file_name)
            self.stack_str_in_QListWidget(self.ui.list_of_models, name)
            self.imported_weights[name] = file_name

    def extract_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select a video file",
                                                   "",
                                                   "(*.mat);;(*.npy);;(*.tiff);;(*.h5)",
                                                   options=QFileDialog.Options())
        if file_name:
            ext = get_file_extension(file_name)
            name = get_file_name(file_name)
            hsi = HSImage()
            if ext == ".mat":
                keys = request_keys_from_mat_file(file_name)
                key = self.show_dialog_with_choice(keys, "Select a key", "Keys:")
                if not key: return
                hsi.load_from_mat(file_name, key)
            elif ext == ".h5":
                keys = request_keys_from_h5_file(file_name)
                key = self.show_dialog_with_choice(keys, "Select a key", "Keys:")
                if not key: return
                hsi.load_from_h5(file_name, key)
            elif ext == ".npy":
                hsi.load_from_npy(file_name)
            elif ext == ".tiff":
                hsi.load_from_tiff(file_name)

            self.stack_str_in_QListWidget(self.ui.extracted_list, name)
            self.current_data[name] = {"hsi": hsi,
                                       "mask": None,
                                       "palette": None,
                                       "shape": hsi.data.shape,
                                       "predict": None,
                                       "confusion_matrix": None,
                                       "classification_report": None}

    def add_mask(self):
        item = self.ui.extracted_list.currentItem()
        if item:
            item = item.text()

            file_name, _ = QFileDialog.getOpenFileName(self,
                                                       "Select a mask file",
                                                       "",
                                                       "(*.npy);;(*.mat);;(*.png);;(*.bmp);;(*.tiff);;(*.h5)",
                                                       options=QFileDialog.Options())
            if file_name:
                mask = self.extract_mask(file_name)
                if mask.data.shape[:2] != self.current_data[item]["shape"][:2]:
                    self.show_error("Mask shape does not match with HSI shape.")
                    return

                # set bold font for item
                self.ui.extracted_list.currentItem().setFont(QFont("Arial", 12, QFont.Bold))
                self.current_data[item]["mask"] = mask
                self.set_pallete_for_key(item, mask.data.shape[2])

    def show_image(self):
        item = self.ui.extracted_list.currentItem()
        if item:
            self.ui.image_view_box.setCurrentIndex(0)
            item = item.text()
            self.current_key = item
            hsi_array = self.current_data[item]["hsi"]
            self.current_image = hsi_array.data
            self.current_hsi = hsi_array
            self.current_predict = self.current_data[item]["predict"]
            self.current_mask = self.current_data[item]["mask"]
            self.ui.spin_channels.setValue(0)
            self.ui.spin_channels.setMaximum(self.current_image.shape[2] - 1)
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.highcontast_check.isChecked(),
                                      0,
                                      self.ui.image_label)

            if (self.current_data[item]["predict"] is not None and
                    self.current_data[item]["confusion_matrix"] is not None and
                    self.current_data[item]["classification_report"] is not None):
                self.current_test_predict = self.current_data[item]["predict"]
                self.current_classification_report = self.current_data[item]["classification_report"]
                self.current_confusion_matrix = self.current_data[item]["confusion_matrix"]

                self.ui.classification_report_label.setText(self.current_classification_report)

    def show_mask(self):
        if self.current_mask is not None:
            self.current_image = convert_to_color_(self.current_mask.get_2d(),
                                                   self.current_data[self.current_key]["palette"])
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.highcontast_check.isChecked(),
                                      self.ui.spin_channels.value(),
                                      self.ui.image_label)

    def show_predict(self):
        if self.current_predict is not None:
            self.current_image = convert_to_color_(self.current_predict,
                                                   self.current_data[self.current_key]["palette"])
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.highcontast_check.isChecked(),
                                      self.ui.spin_channels.value(),
                                      self.ui.image_label)

    def extract_mask(self, file_name):
        mask = HSMask()
        ext = get_file_extension(file_name)
        if ext == ".mat":
            keys = request_keys_from_mat_file(file_name)
            key = self.show_dialog_with_choice(keys, "Select a key", "Keys:")
            if not key: return
            mask.load(file_name, key)

        elif ext == ".h5":
            keys = request_keys_from_h5_file(file_name)
            key = self.show_dialog_with_choice(keys, "Select a key", "Keys:")
            if not key: return
            mask.load(file_name, key)

        else:
            mask.load(file_name)
        return mask

    def set_pallete_for_key(self, key, n_classes):
        palette = get_palette(n_classes)
        self.current_data[key]["palette"] = palette

    def view_changed(self):
        if self.current_image is None: return
        state = self.ui.image_view_box.currentText()
        if state == "Show Mask":
            self.show_mask()
        elif state == "Layers View":
            self.current_image = self.current_hsi.data
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.highcontast_check.isChecked(),
                                      self.ui.spin_channels.value(),
                                      self.ui.image_label)
        elif state == "Show Predict":
            self.show_predict()

    def data_to_test(self):
        self.current_test_predict = None
        self.current_test_hsi = None
        self.current_test_mask = None
        self.current_test_key = None
        self.current_classification_report = None
        self.current_confusion_matrix = None

        self.ui.output_test_data.setText("output_test_data")
        self.ui.predict_test_icon.setText("PREDICT")
        self.ui.hsi_test_icon.setText("HSI")
        self.ui.mask_test_icon.setText("MASK")

        item = self.ui.extracted_list.currentItem()
        if item:
            item = item.text()
            self.current_test_key = item
            self.ui.output_test_data.setText(item)
            self.current_test_hsi = self.current_data[item]["hsi"]
            mid = self.current_test_hsi.data.shape[2] // 2
            self.set_image_to_label(image=self.current_data[item]['hsi'].data,
                                    image_label=self.ui.hsi_test_icon,
                                    scale_factor=20,
                                    high_contrast=True,
                                    channel=mid)

            if self.current_data[item]["mask"]:
                self.current_test_mask = self.current_data[item]["mask"]
                mask_array = self.current_test_mask.get_2d()
                mask_array = convert_to_color_(mask_array, self.current_data[item]["palette"])
                self.set_image_to_label(image=mask_array,
                                        image_label=self.ui.mask_test_icon,
                                        scale_factor=20)

            if isinstance(self.current_data[item]["predict"], np.ndarray):
                self.current_test_predict = self.current_data[item]["predict"]
                mask_array = convert_to_color_(self.current_test_predict, self.current_data[item]["palette"])
                self.set_image_to_label(image=mask_array,
                                        image_label=self.ui.predict_test_icon,
                                        scale_factor=20)

    def data_to_train(self):
        item = self.ui.extracted_list.currentItem()
        if item:
            item = item.text()
            if self.current_data[item]["mask"]:
                self.ui.train_file_output_label.setText(item)
                self.current_train_hsi = self.current_data[item]["hsi"]
                mid = self.current_train_hsi.data.shape[2] // 2
                self.current_train_mask = self.current_data[item]["mask"]
                mask_array = self.current_train_mask.get_2d()
                mask_array = convert_to_color_(mask_array, self.current_data[item]["palette"])
                self.set_image_to_label(image=self.current_data[item]['hsi'].data,
                                        image_label=self.ui.hsi_icon_label,
                                        scale_factor=20,
                                        high_contrast=True,
                                        channel=mid)

                self.set_image_to_label(image=mask_array,
                                        image_label=self.ui.mask_icon_label,
                                        scale_factor=20)
            else:
                self.show_error("Mask is not defined")

    def start_learning(self):
        if self.current_train_hsi and self.current_train_mask:
            self.ui.graphics_loss_view.clear()
            self.ui.graphics_acc_view.clear()
            self.ui.learning_progressbar.setValue(0)
            self.ui.learning_progressbar.setMaximum(int(self.ui.epochs_edit.text()))

            self.g_val_accs = []
            self.g_train_accs = []
            self.g_epochs = []
            self.g_val_losses = []
            self.g_train_losses = []

            if self.ui.need_load_weight_checkBox.isChecked():
                item = self.ui.list_of_models.currentItem()
                if item:
                    weights = self.imported_weights[item.text()]
                else:
                    self.show_info("You did not select a weight file. \nTraining will start from scratch")
                    weights = None
            else:
                weights = None

            optimizer_params = {
                "lr": float(self.ui.lr_edit.text()),
                "weight_decay": float(self.ui.weight_decay_edit.text())}

            scheduler_params = {
                "step_size": float(self.ui.step_size_edit.text()),
                "gamma": float(self.ui.gamma_edit.text())}

            scheduler_type = str(self.ui.scheduler_type_edit.currentText()
                                 ) if self.ui.scheduler_type_edit.currentText() != "None" else None

            fit_params = {
                "epochs": int(self.ui.epochs_edit.text()),
                "train_sample_percentage": float(self.ui.train_split_edit.text()),
                "dataloader_mode": str(self.ui.dataloader_mode_edit.currentText()),
                "batch_size": int(self.ui.batch_size_edit.text()),
                "optimizer_params": optimizer_params,
                "scheduler_type": scheduler_type,
                "scheduler_params": scheduler_params}

            start_time = get_date_time()
            run_name = f"{self.ui.choose_model_for_train.currentText()}_{start_time[0]}_{start_time[1]}"

            fits = {"hsi": self.current_train_hsi,
                    "mask": self.current_train_mask,
                    "model": models_dict[str(self.ui.choose_model_for_train.currentText())],
                    "device": self.devices_dict[str(self.ui.device_box2.currentText())],
                    "optimizer_params": optimizer_params,
                    "scheduler_params": scheduler_params,
                    "fit_params": fit_params,
                    "weights": weights,
                    "run_name": run_name}

            self.ui.start_learning_btn.setEnabled(False)
            self.progress_requested.emit(fits)

    def update_train_progress(self, data):
        if "error" in data:
            self.show_error(data["error"])
            self.ui.start_learning_btn.setEnabled(True)
            return

        if "epoch" in data:
            self.g_epochs.append(data["epoch"])
            self.g_val_losses.append(data["val_loss"])
            self.g_train_losses.append(data["train_loss"])
            self.g_val_accs.append(data["val_acc"])
            self.g_train_accs.append(data["train_acc"])

            self.ui.graphics_loss_view.plot(self.g_epochs, self.g_val_losses, pen='r', name='Validation Loss')
            self.ui.graphics_loss_view.plot(self.g_epochs, self.g_train_losses, pen='g', name='Train Loss')
            self.ui.graphics_acc_view.plot(self.g_epochs, self.g_val_accs, pen='r', name='Validation Accuracy')
            self.ui.graphics_acc_view.plot(self.g_epochs, self.g_train_accs, pen='g', name='Train Accuracy')
            self.ui.learning_progressbar.setValue(data["epoch"])

            if data["model"] != "TF2DCNN":
                net = data["model"]
                torch.save(net.model.state_dict(), f"checkpoints/{data['run_name']}.pth")

                if data["run_name"] not in self.imported_weights:
                    self.stack_str_in_QListWidget(self.ui.list_of_models, data["run_name"])

                self.imported_weights[data["run_name"]] = f"checkpoints/{data['run_name']}.pth"


        if "end" in data:
            self.ui.start_learning_btn.setEnabled(True)
            return

    def change_state_btn_cause_checkbox(self, btn):
        if self.ui.need_load_weight_checkBox.isChecked():
            btn.setEnabled(True)
        else:
            btn.setEnabled(False)

    def stop_train(self):
        if not self.ui.start_learning_btn.isEnabled():
            global stop_train
            stop_train = True

    def start_inference(self):
        item = self.ui.list_of_models.currentItem()
        if self.current_test_hsi and item:
            weight_path = self.imported_weights[item.text()]
            fits = {"hsi": self.current_test_hsi,
                    "weights": weight_path,
                    "device": str(self.ui.device_box.currentText()),
                    "model": models_dict[str(self.ui.choose_model_for_inference.currentText())]}

            self.ui.start_inference_btn.setEnabled(False)
            self.inference_requested.emit(fits)

    def update_inference_progress(self, data):
        if "error" in data:
            self.show_error(data["error"])

        elif "predict" in data:
            self.current_test_predict = data["predict"]
            self.current_data[self.current_test_key]["predict"] = self.current_test_predict
            mask_array = convert_to_color_(self.current_test_predict, self.current_data[self.current_test_key]["palette"])
            self.set_image_to_label(image=mask_array,
                                    image_label=self.ui.predict_test_icon,
                                    scale_factor=20)
        self.ui.start_inference_btn.setEnabled(True)

    def estimate(self):
        if self.current_test_predict is not None and self.current_test_mask is not None:
            pred = self.current_test_predict.ravel()
            gt = self.current_test_mask.get_2d().ravel()
            current_classification_report = classification_report(gt, pred, output_dict=True)
            ncm = confusion_matrix(gt, pred, normalize='true')
            self.current_data[self.current_test_key]["classification_report"] = current_classification_report
            self.current_data[self.current_test_key]["confusion_matrix"] = ncm

            plt_ncm = draw_confusion_matrix(ncm)
            # create numpy array from plt_ncm
            plt_ncm.savefig('temp.png')
            plt_ncm.close()
            ncm_numpy_array = plt.imread('temp.png')
            # convert rgba to rgb
            ncm_numpy_array = ncm_numpy_array[:, :, :3]
            self.current_confusion_matrix = ncm_numpy_array
            self.update_cm()
            self.update_metrics(current_classification_report)

    def update_cm(self):
        if self.current_confusion_matrix is not None:
            self.set_image_to_label(self.current_confusion_matrix, self.ui.cm_matrix_label, int(self.ui.cm_slider.value()))

    def update_metrics(self, report):
        self.ui.accuracy_label.setText(str(report["accuracy"]))
        self.ui.recall_macro_label.setText(str(report["macro avg"]["recall"]))
        self.ui.recall_weighted_label.setText(str(report["weighted avg"]["recall"]))
        self.ui.precis_macro_label.setText(str(report["macro avg"]["precision"]))
        self.ui.precis_weighted_label.setText(str(report["weighted avg"]["precision"]))
        self.ui.f1_macro_label.setText(str(report["macro avg"]["f1-score"]))
        self.ui.f1_weighted_label.setText(str(report["weighted avg"]["f1-score"]))


def draw_confusion_matrix(cm, cmap="Blues"):
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0.0:
                plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="black")
    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    return plt


if __name__ == '__main__':
    create_dir_if_not_exist("checkpoints")
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
