import copy
import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from typing import Optional
from openhsl.hs_device import HSDevice
import openhsl.hs_image_utils as hsiutils

class HSDeviceQt(QObject, HSDevice):
    send_slit_image = pyqtSignal(QImage)
    send_slit_preview_image = pyqtSignal(QImage)
    compute_slit_angle_finished = pyqtSignal()
    # send_slit_angle = pyqtSignal(float)
    # send_slit_offset = pyqtSignal(float)
    def __init__(self):
        super(HSDeviceQt, self).__init__()

        self.slit_image: Optional[np.ndarray] = None
        self.slit_image_to_send: Optional[np.ndarray] = None
        self.row_count = 0
        self.col_count = 0
        self.threshold_value = 40
        self.threshold_type = 0

    @pyqtSlot(str)
    def on_read_slit_image(self, path: str):
        self.slit_image = cv.imread(path, cv.IMREAD_ANYCOLOR)

        if len(self.slit_image.shape) == 2:
            self.slit_image_to_send = cv.cvtColor(self.slit_image, cv.COLOR_GRAY2RGB)
        else:
            self.slit_image = cv.cvtColor(self.slit_image, cv.COLOR_BGR2RGB)
            self.slit_image_to_send = copy.deepcopy(self.slit_image)
        self.col_count, self.row_count = self.slit_image.shape[0:2]
        image_to_draw_qt = QImage(self.slit_image_to_send, self.slit_image_to_send.shape[1],
                                  self.slit_image_to_send.shape[0], QImage.Format.Format_RGB888)
        # QImage references ndarray data, so we need to copy QImage
        # See: https://stackoverflow.com/a/49516303
        self.send_slit_image.emit(image_to_draw_qt.copy())

    @pyqtSlot(QRectF)
    def on_compute_slit_angle(self, area_rect: QRectF):
        x, y, w, h = area_rect.topLeft().x(), area_rect.topLeft().y(), area_rect.width(), area_rect.height()
        self.roi.slit_slope, self.roi.slit_angle, self.roi.slit_intercept = \
            hsiutils.compute_slit_angle(self.slit_image, int(x), int(y), int(w), int(h),
                                        self.threshold_value, self.threshold_type)
        self.compute_slit_angle_finished.emit()

    def on_threshold_slit_image(self):
        image_thresholded = hsiutils.threshold_image(copy.deepcopy(self.slit_image_to_send), self.threshold_value, 255,
                                                     self.threshold_type)
        image_thresholded_qt = QImage(image_thresholded, image_thresholded.shape[1], image_thresholded.shape[0],
                                      QImage.Format.Format_RGB888)
        self.send_slit_preview_image.emit(image_thresholded_qt.copy())
