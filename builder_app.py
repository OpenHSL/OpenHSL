import json
import sys

from copy import deepcopy

from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QLineEdit

from gui.common_gui import CIU
from gui.mac_builder_gui import Ui_MainWindow
from gui.utils import (get_date_time, save_json_dict, create_dir_if_not_exist,
                       get_file_directory, get_file_extension, get_file_name,
                       request_keys_from_mat_file, request_keys_from_h5_file)

from openhsl.base.hsi import HSImage, hsi_to_rgb
from openhsl.build.builder import HSBuilder


class Worker(QObject):
    meta_data = Signal(dict)

    @Slot(dict)
    def do_work(self, meta):
        try:
            hsb = HSBuilder(path_to_data=meta["data"],
                            data_type=meta["data_type"],
                            path_to_gps=meta["telemetry"],
                            path_to_metadata=meta["metadata"])

            hsb.build(principal_slices=meta["principal_slices"],
                      norm_rotation=meta["norm_rotation"],
                      flip_wavelengths=meta["flip_wavelength"],
                      roi=meta["roi"],
                      light_norm=meta["light_normalize"],
                      barrel_dist_norm=meta["barrel_distortion_normalize"])
            meta["date"], meta["time"] = get_date_time()
            meta["hsi"] = hsb.get_hsi()

            self.meta_data.emit(meta)
        except Exception as e:
            meta = {"error": str(e)}
            self.meta_data.emit(meta)


class MainWindow(CIU):
    meta_requested = Signal(dict)

    def __init__(self):
        CIU.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.hsis = {}
        self.pixel_color = None
        self.current_pixel = None
        self.current_hsi = None

        # THREAD SETUP
        self.worker = Worker()
        self.worker_thread = QThread()

        self.worker.meta_data.connect(self.write_meta)
        self.meta_requested.connect(self.worker.do_work)

        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.start()

        # BUTTONS CONNECTIONS
        self.ui.build_file_btn.clicked.connect(self.start_build_from_file)
        self.ui.build_dir_btn.clicked.connect(lambda: self.start_build_from_file(dir=True))
        self.ui.show_btn.clicked.connect(self.show_local_hsi)
        self.ui.save_as_btn.clicked.connect(self.save_hsi)
        self.ui.delete_hsi.clicked.connect(self.delete_hsi_from_hsi_Qlist)
        self.ui.import_hsi.clicked.connect(self.import_hsi)
        self.ui.white_point_btn.clicked.connect(self.set_white_point)

        # OHTER INTERACT
        self.ui.horizontalSlider.valueChanged.connect(
            lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                              self.ui.check_high_contast.isChecked(),
                                              self.ui.spinBox.value(),
                                              self.ui.image_label))

        self.ui.spinBox.valueChanged.connect(lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                                                               self.ui.check_high_contast.isChecked(),
                                                                               self.ui.spinBox.value(),
                                                                               self.ui.image_label))
        self.ui.check_high_contast.stateChanged.connect(
            lambda: self.update_current_image(self.ui.horizontalSlider.value(),
                                              self.ui.check_high_contast.isChecked(),
                                              self.ui.spinBox.value(),
                                              self.ui.image_label))

        self.ui.metadata_checkbox.stateChanged.connect(
            lambda: self.change_state_btn_cause_checkbox(self.ui.save_wavelengths_checkbox))

        self.ui.metadata_checkbox.stateChanged.connect(
            lambda: self.change_state_btn_cause_checkbox(self.ui.check_roi))

        self.ui.metadata_checkbox.stateChanged.connect(
            lambda: self.change_state_btn_cause_checkbox(self.ui.check_light_norm))

        self.ui.view_box.currentIndexChanged.connect(self.view_changed)

        self.show()

    def set_white_point(self):
        if self.current_hsi is None: return
        if self.current_pixel is None: return
        new_hsi = deepcopy(self.current_hsi)
        hyperpixel = new_hsi.data[self.current_pixel[1], self.current_pixel[0], :]
        new_hsi.calibrate_white_reference(hyperpixel)
        time = get_date_time()[1]
        self.hsis[f"{time}_white_point"] = {"hsi": new_hsi}
        self.stack_str_in_QListWidget(self.ui.hsi_Qlist, f"{time}_white_point")

    def view_changed(self):
        if self.current_image is None: return
        state = self.ui.view_box.currentText()
        if state == "layers":
            self.current_image = self.current_hsi.data
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.check_high_contast.isChecked(),
                                      self.ui.spinBox.value(),
                                      self.ui.image_label)
        elif state == "RGB":
            self.current_image = hsi_to_rgb(self.current_hsi)
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      False,
                                      0,
                                      self.ui.image_label)

    def import_hsi(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "(*.npy);;(*.mat);;(*.h5);;(*.tiff)",
                                                   options=QFileDialog.Options())
        if file_name:
            hsi = HSImage()
            name = self.cut_path_with_deep(file_name, 1)
            ext = get_file_extension(file_name)
            if ext == ".mat":
                keys = request_keys_from_mat_file(file_name)
                key = self.show_dialog_with_choice(keys, "Choose key", "Choose key from list")
                if not key: return
                hsi.load_from_mat(file_name, key)

            elif ext == ".h5":
                keys = request_keys_from_h5_file(file_name)
                key = self.show_dialog_with_choice(keys, "Choose key", "Choose key from list")
                if not key: return
                hsi.load_from_h5(file_name, key)

            elif ext == ".npy":
                hsi.load_from_npy(file_name)

            elif ext == ".tiff":
                hsi.load_from_tiff(file_name)

            self.hsis[name] = {"hsi": hsi}
            self.stack_str_in_QListWidget(self.ui.hsi_Qlist, name)

    def start_build_from_file(self, dir=False):
        if dir:
            file_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "(*.avi)",
                                                       options=QFileDialog.Options())
        if file_name:
            meta = {"data": file_name,
                    "norm_rotation": self.ui.check_norm_rotation.isChecked(),
                    "barrel_distortion_normalize": self.ui.check_barrel_dist_norm.isChecked(),
                    "light_normalize": self.ui.check_light_norm.isChecked(),
                    "roi": True if self.ui.check_roi.isChecked() else self.ui.check_roi.isChecked(),
                    "flip_wavelength": self.ui.check_wavelengths.isChecked(),
                    "data_type": self.ui.data_type_box.currentText(),
                    "metadata": None,
                    "telemetry": None,
                    "principal_slices": False}

            if self.ui.metadata_checkbox.isChecked():
                file_name, _ = QFileDialog.getOpenFileName(self,
                                                           "Metadata file for HSI",
                                                           "add metadata",
                                                           "(*.json)",
                                                           options=QFileDialog.Options())
                if file_name:
                    meta["metadata"] = file_name
                else:
                    return

            if self.ui.telemetry_requiared.isChecked():
                file_name, _ = QFileDialog.getOpenFileName(self,
                                                           "Telemetry file",
                                                           "add telemetry",
                                                           "(*.csv);;(*.json)",
                                                           options=QFileDialog.Options())
                if file_name:
                    meta["telemetry"] = file_name
                else:
                    return

            self.ui.build_file_btn.setEnabled(False)
            self.ui.build_dir_btn.setEnabled(False)
            self.meta_requested.emit(meta)

    def write_meta(self, meta):
        if "error" in meta:
            self.ui.build_file_btn.setEnabled(True)
            self.ui.build_dir_btn.setEnabled(True)
            self.show_error(meta["error"])
            return

        self.hsis[meta["time"]] = meta
        self.stack_str_in_QListWidget(self.ui.hsi_Qlist, meta["time"])
        self.ui.build_file_btn.setEnabled(True)
        self.ui.build_dir_btn.setEnabled(True)

    def show_local_hsi(self):
        item = self.ui.hsi_Qlist.currentItem()
        if item:
            item = item.text()
            hsi = self.hsis[item]["hsi"]
            ### IT's a HARDCODE SHIT!!!!!!!!!!!!!
            import numpy as np
            hsi.wavelengths = np.linspace(320, 920, 103)
            ### IT"S A HARDCODE SHIT END!!!!!!!!!!!
            self.current_hsi = hsi

            self.ui.view_box.removeItem(1)

            if len(hsi.wavelengths) == hsi.data.shape[2]:
                self.ui.view_box.addItem("RGB")


            self.current_image = hsi.data
            self.ui.spinBox.setValue(0)
            self.ui.spinBox.setMaximum(self.current_image.shape[2] - 1)
            self.update_current_image(self.ui.horizontalSlider.value(),
                                      self.ui.check_high_contast.isChecked(),
                                      0,
                                      self.ui.image_label)
            self.ui.image_label.mousePressEvent = self.get_pixel_index

    def get_pixel_index(self, event):
        x = event.pos().x()
        y = event.pos().y()
        pixmap = self.ui.image_label.pixmap()
        if pixmap:
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                x = int(x / pixmap.width() * self.current_image.shape[1])
                y = int(y / pixmap.height() * self.current_image.shape[0])
                self.pixel_color = self.current_hsi.data[y, x, :]
                self.current_pixel = (x, y)
                self.update_histogram()

    def update_histogram(self):
        if self.pixel_color is not None:
            self.ui.histogramm.clear()
            self.ui.histogramm.plot(self.pixel_color, pen=(255, 255, 255))

    def save_hsi(self):
        item = self.ui.hsi_Qlist.currentItem()
        if item:
            item = item.text()
            hsi = self.hsis[item]["hsi"]
            meta = self.hsis[item]
            del meta["hsi"]
            file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                       "(*.npy);;(*.mat);;(*.h5);;(*.png);;(*.jpg);;(*.jpeg);;(*.tiff);;(*.bmp)",
                                                       options=QFileDialog.Options())

            if file_name:
                ext = get_file_extension(file_name)
                name = get_file_name(file_name)
                dir = f"{get_file_directory(file_name)}/{name}"
                create_dir_if_not_exist(dir)

                file_name = f"{dir}/hsi{ext}"
                meta_name = f"{dir}/builder_metadata.json"
                save_json_dict(meta, meta_name)

                if ext == ".npy":
                    hsi.save_to_npy(file_name)
                elif ext == ".mat":
                    key, _ = QInputDialog.getText(self, "Input Dialog", "Enter key:", QLineEdit.Normal)
                    if not key:
                        return
                    hsi.save_to_mat(file_name, mat_key=key)
                elif ext == ".h5":
                    key, _ = QInputDialog.getText(self, "Input Dialog", "Enter key:", QLineEdit.Normal)
                    if not key:
                        return
                    hsi.save_to_h5(file_name, h5_key=key)
                elif ext == ".tiff":
                    hsi.save_to_tiff(file_name)
                elif ext == ".png":
                    hsi.save_to_images(file_name, format="png")
                elif ext == ".jpg":
                    hsi.save_to_images(file_name, format="jpg")
                elif ext == ".jpeg":
                    hsi.save_to_images(file_name, format="jpeg")
                elif ext == ".bmp":
                    hsi.save_to_images(file_name, format="bmp")

                if meta["metadata"] is not None and self.ui.save_wavelengths_checkbox.isChecked():
                    meta_name = f"{dir}/hsi_metainfo.json"
                    with open(meta["metadata"], 'r') as json_file:
                        data = json.load(json_file)
                        data = {"wavelengths": data["wavelengths"]}
                    save_json_dict(data, meta_name)

    def delete_hsi_from_hsi_Qlist(self):
        item = self.ui.hsi_Qlist.currentItem()
        if item:
            item = item.text()
            del self.hsis[item]
            self.ui.hsi_Qlist.takeItem(self.ui.hsi_Qlist.currentRow())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
