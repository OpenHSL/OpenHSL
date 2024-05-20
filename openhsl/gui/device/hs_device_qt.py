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
    send_bd_slit_image_contrasted = pyqtSignal(QImage)
    send_bd_distortion_grid_image = pyqtSignal(QImage)
    send_bd_undistorted_slit_image = pyqtSignal(QImage)

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
        self.bd_contrast_value = 2
        self.bd_slit_contrasted: Optional[np.ndarray] = None

        # Current wavelength
        self.wl_slit_image: Optional[np.ndarray] = None
        self.wl_slit_image_to_send: Optional[np.ndarray] = None
        self.wl_data = HSCalibrationWavelengthData()

    def get_equation_params(self) -> Dict[str, List[float]]:
        return self.calib_slit_data.barrel_distortion_params

    def get_slit_angle_min(self):
        return self.slit_angle_min

    def get_slit_angle_max(self):
        return self.slit_angle_max

    def get_slit_intercept_min(self):
        return self.slit_intercept_min

    def get_slit_intercept_max(self):
        return self.slit_intercept_max

    def set_equation_params(self, barrel_distortion_params: Dict[str, List[float]]):
        self.calib_slit_data.barrel_distortion_params = barrel_distortion_params

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

    def is_equation_data_enough(self):
        enough = False
        if self.calib_slit_data.barrel_distortion_params is not None:
            enough = len(self.calib_slit_data.barrel_distortion_params['powers']) > 0 and \
                     len(self.calib_slit_data.barrel_distortion_params['coeffs']) > 0 and \
                     len(self.calib_slit_data.barrel_distortion_params['factors']) > 0
        return enough

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
    def on_contrast_bd_slit_image(self):
        self.bd_slit_contrasted = hsiutils.contrast_image(copy.deepcopy(self.slit_image_rotated_rgb), self.bd_contrast_value)
        image_contrasted_qt = QImage(self.bd_slit_contrasted,
                                     self.bd_slit_contrasted.shape[1], self.bd_slit_contrasted.shape[0],
                                     QImage.Format.Format_RGB888)
        self.send_bd_slit_image_contrasted.emit(image_contrasted_qt.copy())

    @pyqtSlot(bool)
    def on_draw_distortion_grid(self, contrasted: bool = False):
        factors = np.array(self.calib_slit_data.barrel_distortion_params['factors'])
        coeffs = np.array(self.calib_slit_data.barrel_distortion_params['coeffs'])
        powers = np.array(self.calib_slit_data.barrel_distortion_params['powers'])
        image = self.slit_image_rotated_rgb
        if contrasted:
            image = self.bd_slit_contrasted
        image_with_distortion_grid = utils.apply_barrel_distortion(copy.deepcopy(image), coeffs, powers, factors)
        image_with_distortion_grid_qt = QImage(image_with_distortion_grid,
                                               image_with_distortion_grid.shape[1], image_with_distortion_grid.shape[0],
                                               QImage.Format.Format_RGB888)
        self.send_bd_distortion_grid_image.emit(image_with_distortion_grid_qt.copy())

    @pyqtSlot(bool)
    def on_undistort_slit_image(self, contrasted: bool = False):
        factors = np.array(self.calib_slit_data.barrel_distortion_params['factors'])
        coeffs = np.array(self.calib_slit_data.barrel_distortion_params['coeffs'])
        powers = np.array(self.calib_slit_data.barrel_distortion_params['powers'])
        center_xy = np.array(self.calib_slit_data.barrel_distortion_params['center'])
        image = self.slit_image_rotated_rgb
        if contrasted:
            image = self.bd_slit_contrasted
        undistortion_coeffs = \
            list(hsiutils.compute_undistortion_coeffs(coeffs, powers, factors, image.shape, center_xy))
        self.calib_slit_data.barrel_distortion_params['undistortion_coeffs'] = undistortion_coeffs
        image_undistorted = hsiutils.undistort_image(image, undistortion_coeffs, list(center_xy))
        image_undistorted_qt = QImage(image_undistorted,
                                      image_undistorted.shape[1], image_undistorted.shape[0],
                                      QImage.Format.Format_RGB888)
        self.send_bd_undistorted_slit_image.emit(image_undistorted_qt.copy())
