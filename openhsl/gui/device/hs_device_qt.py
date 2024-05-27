import copy
import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from typing import Dict, List, Optional, Tuple, Union
from openhsl.build.hs_device import HSDevice, HSCalibrationWavelengthData
import openhsl.build.hs_image_utils as hsiutils
import openhsl.build.utils as butils
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
    send_bd_slit_center = pyqtSignal(int, int)
    send_wl_image = pyqtSignal(QImage, str)
    send_wl_image_count = pyqtSignal(int)

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
        self.bd_grid_tile_size = 40

        # Current wavelength
        self.wl_image_path_list: List[str] = None
        self.wl_image: Optional[np.ndarray] = None
        self.wl_image_preview: Optional[np.ndarray] = None
        self.wl_contrast_value = 2
        self.wl_image_to_send: Optional[np.ndarray] = None
        self.wl_data = HSCalibrationWavelengthData()

    def get_slit_angle_min(self):
        return self.slit_angle_min

    def get_slit_angle_max(self):
        return self.slit_angle_max

    def get_slit_intercept_min(self):
        return self.slit_intercept_min

    def get_slit_intercept_max(self):
        return self.slit_intercept_max

    def get_slit_intercept_rotated(self):
        return np.rint(0.5 * self.get_slit_slope() * self.slit_image_width + self.get_slit_intercept())

    def set_bd_contrast_value(self, value: float):
        self.bd_contrast_value = value

    def set_center(self, center_x: int, center_y: int):
        self.calib_slit_data.barrel_distortion_params['center'] = [center_x, center_y]

    def set_barrel_distortion_params(self, barrel_distortion_params: Dict[str, Union[List[float], List[int]]]):
        self.calib_slit_data.barrel_distortion_params = barrel_distortion_params

    def set_wl_contrast_value(self, value: float):
        self.wl_contrast_value = value

    def set_grid_tile_size(self, value: int):
        self.bd_grid_tile_size = value

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

    def set_wavelength_calibration_data(self, data: np.ndarray):
        if data.shape[1] == 3:
            self.calib_wavelength_data.wavelength_list = data[:, 0].tolist()
            self.calib_wavelength_data.wavelength_y_list = data[:, 1].tolist()
            self.calib_wavelength_data.wavelength_slit_offset_y_list = data[:, 2].tolist()

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

    @staticmethod
    def image_to_qimage(image: np.ndarray, image_format: QImage.Format = QImage.Format.Format_RGB888):
        image_qt = QImage(image, image.shape[1], image.shape[0], image_format)
        return image_qt

    def is_center_defined(self) -> bool:
        defined = False
        if self.calib_slit_data.barrel_distortion_params is not None:
            center_xy = self.calib_slit_data.barrel_distortion_params['center']
            if center_xy is not None:
                if len(center_xy) == 2:
                    defined = True
        return defined

    def is_equation_data_enough(self) -> bool:
        enough = False
        if self.calib_slit_data is not None:
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
            self.calib_slit_data.image_shape = self.slit_image.shape[0:2]
            self.slit_image_rotated = copy.deepcopy(self.slit_image_to_send)
            image_to_draw_qt = self.image_to_qimage(self.slit_image_to_send)
            # QImage references ndarray data, so we need to copy QImage
            # See: https://stackoverflow.com/a/49516303
            self.send_slit_image.emit(image_to_draw_qt.copy())

    @pyqtSlot(str)
    def on_read_wl_image_dir(self, dir_path: str):
        self.wl_image_path_list = butils.file_path_list(dir_path, ['jpg', 'png', 'bmp', 'tif'])
        self.send_wl_image_count.emit(len(self.wl_image_path_list))

    @pyqtSlot(int, bool, bool, bool)
    def on_read_wl_image(self, index: int, apply_rotation: bool = False, apply_undistortion: bool = False,
                         apply_contast_preview: bool = False):
        if self.wl_image_path_list is not None:
            image_count = len(self.wl_image_path_list)
            if index < image_count:
                self.wl_image = cv.imread(self.wl_image_path_list[index], cv.IMREAD_ANYCOLOR)
                inage_name = butils.file_complete_name(self.wl_image_path_list[index])
                # TODO send message if None
                if self.wl_image is not None:
                    if len(self.wl_image.shape) == 3:
                        self.wl_image = cv.cvtColor(self.wl_image, cv.COLOR_BGR2RGB)
                    if apply_rotation:
                        self.wl_image = hsiutils.rotate_image(self.wl_image, self.get_slit_angle())
                    if apply_undistortion:
                        # TODO check undistortion coeffs
                        undistortion_coefficients = self.get_undistortion_coefficients()
                        self.wl_image = hsiutils.undistort_image(self.wl_image, undistortion_coefficients)
                    if len(self.wl_image.shape) == 2:
                        self.wl_image_preview = cv.cvtColor(self.wl_image, cv.COLOR_GRAY2RGB)
                    else:
                        self.wl_image_preview = copy.deepcopy(self.wl_image)
                    if apply_contast_preview:
                        self.wl_image_preview = hsiutils.contrast_image(self.wl_image_preview, self.wl_contrast_value)
                    image_qt = self.image_to_qimage(self.wl_image_preview)
                    self.send_wl_image.emit(image_qt.copy(), inage_name)

    @pyqtSlot(QRectF)
    def on_compute_slit_angle(self, area_rect: QRectF):
        x, y, w, h = area_rect.topLeft().x(), area_rect.topLeft().y(), area_rect.width(), area_rect.height()
        self.calib_slit_data.slope, self.calib_slit_data.angle, self.calib_slit_data.intercept = \
            hsiutils.compute_slit_angle(self.slit_image, int(x), int(y), int(w), int(h),
                                        self.threshold_value, self.threshold_type)
        self.calib_slit_data.slit_roi_origin_x = int(x)
        self.calib_slit_data.slit_roi_origin_y = int(y)
        self.calib_slit_data.slit_roi_width = int(w)
        self.calib_slit_data.slit_roi_height = int(h)

        self.compute_slit_angle_adjusting_range()
        self.compute_slit_intercept_adjusting_range()

        self.compute_slit_angle_finished.emit()

    @pyqtSlot()
    def on_threshold_slit_image(self):
        image_thresholded = hsiutils.threshold_image(copy.deepcopy(self.slit_image_to_send), self.threshold_value, 255,
                                                     self.threshold_type)
        image_thresholded_qt = self.image_to_qimage(image_thresholded)
        self.send_slit_preview_image.emit(image_thresholded_qt.copy())

    @pyqtSlot()
    def on_rotate_bd_slit_image(self):
        self.slit_image_rotated = hsiutils.rotate_image(copy.deepcopy(self.slit_image), self.get_slit_angle())
        self.slit_image_rotated_rgb = hsiutils.rotate_image(copy.deepcopy(self.slit_image_to_send),
                                                            self.get_slit_angle())
        slit_image_rotated_qt = self.image_to_qimage(self.slit_image_rotated_rgb)
        self.send_bd_slit_image_rotated.emit(slit_image_rotated_qt.copy())

    @pyqtSlot()
    def on_contrast_bd_slit_image(self):
        self.bd_slit_contrasted = hsiutils.contrast_image(copy.deepcopy(self.slit_image_rotated_rgb), self.bd_contrast_value)
        image_contrasted_qt = self.image_to_qimage(self.bd_slit_contrasted)
        self.send_bd_slit_image_contrasted.emit(image_contrasted_qt.copy())

    @pyqtSlot(bool)
    def on_draw_distortion_grid(self, contrasted: bool = False):
        factors = np.array(self.calib_slit_data.barrel_distortion_params['factors'])
        coeffs = np.array(self.calib_slit_data.barrel_distortion_params['coeffs'])
        powers = np.array(self.calib_slit_data.barrel_distortion_params['powers'])
        center_xy = np.array(self.calib_slit_data.barrel_distortion_params['center'])
        image = self.slit_image_rotated_rgb
        if contrasted:
            image = self.bd_slit_contrasted
        image_with_distortion_grid = utils.apply_barrel_distortion(copy.deepcopy(image),
                                                                   coeffs, powers, factors, center_xy,
                                                                   self.bd_grid_tile_size)
        image_with_distortion_grid_qt = self.image_to_qimage(image_with_distortion_grid)
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
        image_undistorted_qt = self.image_to_qimage(image_undistorted)
        self.send_bd_undistorted_slit_image.emit(image_undistorted_qt.copy())

    @pyqtSlot(QRectF, int)
    def on_compute_bd_slit_center(self, area_rect: QRectF, threshold_value: int):
        image = copy.deepcopy(self.slit_image_rotated)
        x, y, w, h = area_rect.topLeft().x(), area_rect.topLeft().y(), area_rect.width(), area_rect.height()
        x, y, w, h = int(x), int(y), int(w), int(h)
        image_thresholded = hsiutils.threshold_image(image[y:y + h, x:x + w], threshold_value, 255, self.threshold_type)
        points = np.argwhere(image_thresholded > 0)
        center_x = int(points[:, 1].sum() / len(points)) + x
        center_y = int(points[:, 0].sum() / len(points)) + y
        barrel_distortion_params = self.get_barrel_distortion_params()
        if barrel_distortion_params is None:
            barrel_distortion_params = {'center': [], 'powers': [], 'coeffs': [], 'factors': []}
            self.set_barrel_distortion_params(barrel_distortion_params)
        self.calib_slit_data.barrel_distortion_params['center'] = [center_x, center_y]
        self.send_bd_slit_center.emit(center_x, center_y)
