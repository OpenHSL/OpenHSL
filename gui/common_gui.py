from gui.utils import get_high_contrast, scale_image
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QInputDialog
from PyQt5.QtGui import QImage, QPixmap


class CIU(QMainWindow):  # CIU = Common Interface Utilities
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("OpenHSL")
        self.current_image = None

    @staticmethod
    def show_error(message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

    @staticmethod
    def nparray_2_qimage(array):
        if array.dtype != np.uint8:
            array = (array / np.max(array) * 255).astype(np.uint8)

        height, width = array.shape[:2]
        qimage = QImage(array.tobytes(), width, height, width, QImage.Format_Grayscale8)
        return qimage

    @staticmethod
    def nparray_2_qimage_rgb(array):
        if array.dtype != np.uint8:
            array = (array / np.max(array) * 255).astype(np.uint8)

        height, width = array.shape[:2]
        qimage = QImage(array.tobytes(), width, height, width * 3, QImage.Format_RGB888)
        return qimage

    @staticmethod
    def stack_str_in_QListWidget(list_widget, string):
        list_widget.addItem(string)
        list_widget.scrollToBottom()

    def import_file(self, formats: str, destination_widget):
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   formats, options=QFileDialog.Options())
        if file_name:
            self.stack_str_in_QListWidget(destination_widget, file_name)

    def import_dir(self, destination_widget):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_name:
            self.stack_str_in_QListWidget(destination_widget, dir_name)

    @staticmethod
    def delete_from_QListWidget(list_widget):
        list_widget.takeItem(list_widget.currentRow())

    def set_image_to_label(self, image, image_label, scale_factor, high_contrast=True, channel=0):
        mask = image.shape[2] == 3
        if not mask:
            image = image[:, :, channel]
            if high_contrast:
                image = get_high_contrast(image)
        image = scale_image(image, scale_factor)
        if mask:
            q_image = self.nparray_2_qimage_rgb(image)
        else:
            q_image = self.nparray_2_qimage(image)
        image_label.setPixmap(QPixmap(q_image))

    def update_current_image(self,
                             scale_factor: int,
                             high_contrast: bool,
                             channel: int,
                             image_label):
        if isinstance(self.current_image, np.ndarray):
            mask = self.current_image.shape[2] == 3
            if not mask:
                image = self.current_image[:, :, channel]
                if high_contrast:
                    image = get_high_contrast(image)
            else:
                image = self.current_image
            if scale_factor != 100:
                image = scale_image(image, scale_factor)

            if not mask:
                q_image = self.nparray_2_qimage(image)
            else:
                q_image = self.nparray_2_qimage_rgb(image)
            image_label.setPixmap(QPixmap(q_image))

    def show_dialog_with_choice(self, items, title, message):
        item, ok = QInputDialog.getItem(self, title, message, items)
        if ok and item:
            return item
        else:
            return None

    @staticmethod
    def change_state_btn_cause_checkbox(btn):
        if btn.isEnabled():
            btn.setEnabled(False)
        else:
            btn.setEnabled(True)

    @staticmethod
    def cut_path_with_deep(path, deep):
        path = path.split("/")
        path = path[-deep:]
        path = "/".join(path)
        return path
