import copy
import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from typing import Dict, List, Optional
from openhsl.build.hs_device import HSDevice, HSCalibrationWavelengthData
import openhsl.build.hs_image_utils as hsiutils
import openhsl.gui.device.utils as utils


class HSDeviceQt(QObject, HSDevice):
    send_slit_image = pyqtSignal(QImage)
    send_slit_preview_image = pyqtSignal(QImage)
    compute_slit_angle_finished = pyqtSignal()
    adjust_slit_angle_range = pyqtSignal(float, float)
    adjust_slit_intercept_range = pyqtSignal(float, float)
    send_bd_slit_image_rotated = pyqtSignal(QImage)
    send_bd_slit_image_thresholded = pyqtSignal(QImage)
    send_bd_slit_image_edged = pyqtSignal(QImage, np.ndarray)

    # send_slit_angle = pyqtSignal(float)
    # send_slit_offset = pyqtSignal(float)
    def __init__(self):
        super(HSDeviceQt, self).__init__()

        self.slit_image: Optional[np.ndarray] = None
        self.slit_image_to_send: Optional[np.ndarray] = None
        self.slit_image_height = 0
        self.slit_image_width = 0
        self.threshold_value = 40
        self.threshold_type = 0

        self.slit_angle_range = 1
        self.slit_angle_min = 0
        self.slit_angle_max = 0
        self.slit_intercept_min = 0
        self.slit_intercept_max = 0

        self.slit_image_rotated: Optional[np.ndarray] = None
        self.slit_image_rotated_rgb: Optional[np.ndarray] = None
        self.bd_threshold_value = 40
        self.bd_slit_image_edged: Optional[np.ndarray] = None
        self.bd_slit_image_edged_roi: Optional[np.ndarray] = None
        self.bd_sobel_kernel_size = 5
        self.bd_corners: Optional[np.ndarray] = None
        self.bd_coeffs_dict: Optional[Dict[str, List[float]]] = None

        # Current wavelength
        self.wl_slit_image: Optional[np.ndarray] = None
        self.wl_slit_image_to_send: Optional[np.ndarray] = None
        self.wl_data = HSCalibrationWavelengthData()

    def get_spectrum_corners(self):
        return copy.deepcopy(self.bd_corners)

    def get_slit_angle_min(self):
        return self.slit_angle_min

    def get_slit_angle_max(self):
        return self.slit_angle_max

    def get_slit_intercept_min(self):
        return self.slit_intercept_min

    def get_slit_intercept_max(self):
        return self.slit_intercept_max

    def set_equation_params(self, coeffs_dict: Dict[str, List[float]]):
        self.bd_coeffs_dict = coeffs_dict

    def set_slit_angle(self, value: float):
        self.calib_slit_data.angle = value
        self.calib_slit_data.slope = np.tan(value * np.pi / 180)
        self.compute_slit_angle_adjusting_range()
        self.compute_slit_intercept_adjusting_range()

        # Slit angle priority change is higher
        if self.calib_slit_data.intercept < self.slit_intercept_min:
            self.calib_slit_data.intercept = self.slit_intercept_min
        elif self.calib_slit_data.intercept > self.slit_intercept_max:
            self.calib_slit_data.intercept = self.slit_intercept_max

    def set_slit_intercept(self, value: float):
        self.calib_slit_data.intercept = value
        self.compute_slit_intercept_adjusting_range()

    def compute_slit_angle_adjusting_range(self):
        # Compute min and max angle adjusting range for GUI
        self.slit_angle_min = np.floor(self.calib_slit_data.angle) - self.slit_angle_range
        self.slit_angle_max = np.ceil(self.calib_slit_data.angle) + self.slit_angle_range

        self.adjust_slit_angle_range.emit(self.slit_angle_min, self.slit_angle_max)

    def compute_slit_intercept_adjusting_range(self):
        # Compute min and max intercept for GUI
        limits = []
        if self.get_slit_slope() < 0:
            limits = [-self.get_slit_slope() * self.slit_image_width, self.slit_image_height]
        else:
            limits = [0, self.slit_image_height - self.get_slit_slope() * self.slit_image_width]
        limits = np.rint(limits).astype(int).tolist()
        self.slit_intercept_min, self.slit_intercept_max = limits

        self.adjust_slit_intercept_range.emit(self.slit_intercept_min, self.slit_intercept_max)

    @pyqtSlot(str)
    def on_read_slit_image(self, path: str):
        self.slit_image = cv.imread(path, cv.IMREAD_ANYCOLOR)

        if self.slit_image is not None:
            if len(self.slit_image.shape) == 2:
                self.slit_image_to_send = cv.cvtColor(self.slit_image, cv.COLOR_GRAY2RGB)
            else:
                self.slit_image = cv.cvtColor(self.slit_image, cv.COLOR_BGR2RGB)
                self.slit_image_to_send = copy.deepcopy(self.slit_image)
            self.slit_image_width, self.slit_image_height = self.slit_image.shape[0:2]
            self.slit_image_rotated = copy.deepcopy(self.slit_image_to_send)
            image_to_draw_qt = QImage(self.slit_image_to_send, self.slit_image_to_send.shape[1],
                                      self.slit_image_to_send.shape[0], QImage.Format.Format_RGB888)
            # QImage references ndarray data, so we need to copy QImage
            # See: https://stackoverflow.com/a/49516303
            self.send_slit_image.emit(image_to_draw_qt.copy())

    @pyqtSlot(QRectF)
    def on_compute_slit_angle(self, area_rect: QRectF):
        x, y, w, h = area_rect.topLeft().x(), area_rect.topLeft().y(), area_rect.width(), area_rect.height()
        self.calib_slit_data.slope, self.calib_slit_data.angle, self.calib_slit_data.intercept = \
            hsiutils.compute_slit_angle(self.slit_image, int(x), int(y), int(w), int(h),
                                        self.threshold_value, self.threshold_type)
        self.calib_slit_data.x = int(x)
        self.calib_slit_data.y = int(y)
        self.calib_slit_data.width = int(w)
        self.calib_slit_data.height = int(h)

        self.compute_slit_angle_adjusting_range()
        self.compute_slit_intercept_adjusting_range()

        self.compute_slit_angle_finished.emit()

    @pyqtSlot()
    def on_threshold_slit_image(self):
        image_thresholded = hsiutils.threshold_image(copy.deepcopy(self.slit_image_to_send), self.threshold_value, 255,
                                                     self.threshold_type)
        image_thresholded_qt = QImage(image_thresholded, image_thresholded.shape[1], image_thresholded.shape[0],
                                      QImage.Format.Format_RGB888)
        self.send_slit_preview_image.emit(image_thresholded_qt.copy())

    @pyqtSlot()
    def on_rotate_bd_slit_image(self):
        self.slit_image_rotated = hsiutils.rotate_image(copy.deepcopy(self.slit_image), self.get_slit_angle())
        self.slit_image_rotated_rgb = hsiutils.rotate_image(copy.deepcopy(self.slit_image_to_send),
                                                            self.get_slit_angle())
        slit_image_rotated_qt = QImage(self.slit_image_rotated_rgb,
                                       self.slit_image_rotated_rgb.shape[1], self.slit_image_rotated_rgb.shape[0],
                                       QImage.Format.Format_RGB888)
        self.send_bd_slit_image_rotated.emit(slit_image_rotated_qt.copy())

    @pyqtSlot()
    def on_threshold_bd_slit_image(self):
        image_thresholded = hsiutils.threshold_image(copy.deepcopy(self.slit_image_rotated_rgb),
                                                     self.bd_threshold_value, 255, self.threshold_type)
        image_thresholded_qt = QImage(image_thresholded, image_thresholded.shape[1], image_thresholded.shape[0],
                                      QImage.Format.Format_RGB888)
        self.send_bd_slit_image_thresholded.emit(image_thresholded_qt.copy())

    @pyqtSlot(QRectF)
    def on_edge_bd_slit_image(self, rect: QRectF):
        self.bd_slit_image_edged_roi = np.rint([rect.topLeft().y(), rect.topLeft().x(),
                                                rect.height(), rect.width()]).astype(int)
        image_thresholded = hsiutils.threshold_image(copy.deepcopy(self.slit_image_rotated_rgb),
                                                     self.bd_threshold_value, 255, self.threshold_type)
        image_edged, self.bd_corners = utils.detect_corners(image_thresholded, self.bd_slit_image_edged_roi,
                                                            self.bd_sobel_kernel_size)
        self.bd_corners = np.array(self.bd_corners)
        image_edged = cv.cvtColor(image_edged.astype(np.float32), cv.COLOR_GRAY2RGB)
        image_edged[image_edged > 0] = 1
        image_edged = (image_edged * 255.0).astype(np.uint8)
        image_edged_qt = QImage(image_edged, image_edged.shape[1], image_edged.shape[0], QImage.Format.Format_RGB888)
        self.send_bd_slit_image_edged.emit(image_edged_qt.copy(), self.bd_corners)
