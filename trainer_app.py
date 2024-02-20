import sys
import torch.nn as nn
from tqdm import trange

from PyQt5.QtWidgets import (QApplication, QFileDialog)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot

from gui.utils import (get_file_name, get_file_extension, get_file_directory,
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
from openhsl.models.tf2dcnn import TF2DCNN

models_dict = {
    "M1DCNN": M1DCNN,
    "M3DCNN_li": LI,
    "NM3DCNN": NM3DCNN,
    "SSFTT": SSFTT,
    "TF2DCNN": TF2DCNN
}

stop_train = False


class TrainWorker(QObject):
    progress_signal = Signal(dict)

    @Slot(dict)
    def do_train(self, fits):
        global stop_train
        try:
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

            criterion = nn.CrossEntropyLoss(weight=net.hyperparams["weights"])
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

                # temp_train и val_train можно дергать для графика!
                self.progress_signal.emit({"epoch": e + 1,
                                           "val_loss": temp_val[1],
                                           "train_loss": temp_train['avg_train_loss']})

                if stop_train:
                    stop_train = False
                    break

            self.progress_signal.emit({"end": True})
        except Exception as e:
            self.progress_signal.emit({"error": str(e)})


class MainWindow(CIU):
    progress_requested = Signal(dict)

    def __init__(self):
        CIU.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.current_data = {}
        self.current_hsi = None
        self.current_mask = None
        self.current_predict = None
        self.current_key = None
        self.loaded_weight_for_train = None
        self.loaded_weight_for_inference = None

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

        # THREAD SETUP
        self.train_worker = TrainWorker()
        self.train_thread = QThread()

        self.train_worker.progress_signal.connect(self.update_train_progress)
        self.progress_requested.connect(self.train_worker.do_train)

        self.train_worker.moveToThread(self.train_thread)

        self.train_thread.start()

        self.show()

        # NAVIGATION
        self.ui.trainer_mod_btn.clicked.connect(lambda:
                                                self.ui.stackedWidget.setCurrentWidget(self.ui.trainer_widget))

        self.ui.inference_mod_btn.clicked.connect(lambda:
                                                  self.ui.stackedWidget.setCurrentWidget(self.ui.inference_widget))

        # BUTTON CONNECTIONS
        self.ui.import_data_btn.clicked.connect(self.extract_data)
        self.ui.add_mask_btn.clicked.connect(self.add_mask)
        self.ui.Show.clicked.connect(self.show_image)
        self.ui.data_to_train_btn.clicked.connect(self.data_to_train)
        self.ui.data_to_test_btn.clicked.connect(self.data_to_test)
        self.ui.start_learning_btn.clicked.connect(self.start_learning)
        self.ui.browse_weight_for_train_btn.clicked.connect(self.browse_weight_for_train)
        self.ui.browse_weight_for_inference_btn.clicked.connect(self.browse_weight_for_inference)
        self.ui.stop_train_btn.clicked.connect(self.stop_train)

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
        self.ui.need_load_weight_checkBox.stateChanged.connect(
            lambda: self.change_state_btn_cause_checkbox(self.ui.browse_weight_for_train_btn))

    def extract_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select a video file",
                                                   "",
                                                   "(*.npy);;(*.mat);;(*.tiff);;(*.h5)",
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
            self.ui.graphicsView.clear()
            self.ui.learning_progressbar.setValue(0)
            self.ui.learning_progressbar.setMaximum(int(self.ui.epochs_edit.text()))

            self.g_epochs = []
            self.g_val_losses = []
            self.g_train_losses = []

            if self.ui.need_load_weight_checkBox.isChecked():
                if self.loaded_weight_for_train is None:
                    self.show_error("Weight is not loaded")
                    return

                weights = self.loaded_weight_for_train

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

            fits = {"hsi": self.current_train_hsi,
                    "mask": self.current_train_mask,
                    "model": models_dict[str(self.ui.choose_model_for_inference.currentText())],
                    "device": self.devices_dict[str(self.ui.device_box2.currentText())],
                    "optimizer_params": optimizer_params,
                    "scheduler_params": scheduler_params,
                    "fit_params": fit_params,
                    "weights": weights}

            self.ui.start_learning_btn.setEnabled(False)
            self.progress_requested.emit(fits)

    def update_train_progress(self, data):
        if "error" in data:
            self.show_error(data["error"])
            self.ui.start_learning_btn.setEnabled(True)
            return

        if "end" in data:
            self.ui.start_learning_btn.setEnabled(True)
            return

        self.g_epochs.append(data["epoch"])
        self.g_val_losses.append(data["val_loss"])
        self.g_train_losses.append(data["train_loss"])

        self.ui.graphicsView.plot(self.g_epochs, self.g_val_losses, pen='r', name='Validation Loss')
        self.ui.graphicsView.plot(self.g_epochs, self.g_train_losses, pen='g', name='Train Loss')
        self.ui.learning_progressbar.setValue(data["epoch"])

    def browse_weight_for_train(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select a weight file",
                                                   "",
                                                   "(*.pth);;(*.pt)",
                                                   options=QFileDialog.Options())
        if file_name:
            self.ui.current_label_weight_for_train.setText(self.cut_path_with_deep(file_name, 2))
            self.loaded_weight_for_train = file_name

    def change_state_btn_cause_checkbox(self, btn):
        if self.ui.need_load_weight_checkBox.isChecked():
            btn.setEnabled(True)
        else:
            btn.setEnabled(False)

    def browse_weight_for_inference(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select a weight file",
                                                   "experiment",
                                                   "(*.pth);;(*.pt)",
                                                   options=QFileDialog.Options())
        if file_name:
            self.ui.current_loaded_weight_for_inference.setText(self.cut_path_with_deep(file_name, 2))
            self.loaded_weight_for_inference = file_name

    def stop_train(self):
        if not self.ui.start_learning_btn.isEnabled():
            global stop_train
            stop_train = True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
