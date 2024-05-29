import copy
import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from typing import Dict, List, Optional, Tuple, Union
from openhsl.build.hs_device import HSDevice, HSDeviceType, HSCalibrationWavelengthData
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
    send_ilm_image = pyqtSignal(QImage)

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

        self.ilm_image: Optional[np.ndarray] = None
        self.ilm_image_roi: Optional[np.ndarray] = None
        self.ilm_image_preview: Optional[np.ndarray] = None
        self.ilm_image_roi_preview: Optional[np.ndarray] = None

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

    def get_wavelength_calibration_data(self) -> HSCalibrationWavelengthData:
        return self.wavelength_data

    def set_bd_contrast_value(self, value: float):
        self.bd_contrast_value = value

    def set_center(self, center_x: int, center_y: int):
        self.slit_data.barrel_distortion_params['center'] = [center_x, center_y]

    def set_device_type(self, device_type: HSDeviceType):
        self.device_type = device_type

    def set_barrel_distortion_params(self, barrel_distortion_params: Dict[str, Union[List[float], List[int]]]):
        self.slit_data.barrel_distortion_params = barrel_distortion_params

    def set_wl_contrast_value(self, value: float):
        self.wl_contrast_value = value

    def set_grid_tile_size(self, value: int):
        self.bd_grid_tile_size = value

    def set_slit_angle(self, value: float):
        self.slit_data.angle = value
        self.slit_data.slope = np.tan(value * np.pi / 180)
        self.compute_slit_angle_adjusting_range()
        self.compute_slit_intercept_adjusting_range()

        # Slit angle priority change is higher
        if self.slit_data.intercept < self.slit_intercept_min:
            self.slit_data.intercept = self.slit_intercept_min
        elif self.slit_data.intercept > self.slit_intercept_max:
            self.slit_data.intercept = self.slit_intercept_max

    def set_wavelength_calibration_data(self, wavelength_data: np.ndarray, spectrum_slit_offset_y: int,
                                        spectrum_left_bound: int, spectrum_right_bound: int):
        if wavelength_data.shape[1] == 3:
            self.wavelength_data.wavelength_list = wavelength_data[:, 0].tolist()
            self.wavelength_data.wavelength_y_list = wavelength_data[:, 1].astype(int).tolist()
            self.wavelength_data.wavelength_slit_offset_y_list = wavelength_data[:, 2].astype(int).tolist()
            self.wavelength_data.spectrum_slit_offset_y = spectrum_slit_offset_y
            self.wavelength_data.spectrum_left_bound = spectrum_left_bound
            self.wavelength_data.spectrum_right_bound = spectrum_right_bound

    def set_slit_intercept(self, value: float):
        self.slit_data.intercept = value
        self.compute_slit_intercept_adjusting_range()

    def compute_slit_angle_adjusting_range(self):
        # Compute min and max angle adjusting range for GUI
        self.slit_angle_min = np.floor(self.slit_data.angle) - self.slit_angle_range
        self.slit_angle_max = np.ceil(self.slit_data.angle) + self.slit_angle_range

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
        total_bytes = image.nbytes
        bytes_per_line = int(total_bytes / image.shape[0])
        image_qt = QImage(image, image.shape[1], image.shape[0], bytes_per_line, image_format)
        return image_qt

    def is_center_defined(self) -> bool:
        defined = False
        if self.slit_data.barrel_distortion_params is not None:
            center_xy = self.slit_data.barrel_distortion_params['center']
            if center_xy is not None:
                if len(center_xy) == 2:
                    defined = True
        return defined

    def is_equation_data_enough(self) -> bool:
        enough = False
        if self.slit_data is not None:
            if self.slit_data.barrel_distortion_params is not None:
                enough = len(self.slit_data.barrel_distortion_params['powers']) > 0 and \
                         len(self.slit_data.barrel_distortion_params['coeffs']) > 0 and \
                         len(self.slit_data.barrel_distortion_params['factors']) > 0
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
            self.slit_data.image_shape = self.slit_image.shape[0:2]
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

    @pyqtSlot(str)
    def on_read_ilm_image(self, path: str):
        self.ilm_image = cv.imread(path, cv.IMREAD_ANYCOLOR)

        if self.ilm_image is not None:
            if len(self.ilm_image.shape) == 3:
                self.ilm_image = cv.cvtColor(self.ilm_image, cv.COLOR_BGR2RGB)
            # Apply rotation
            self.ilm_image = hsiutils.rotate_image(self.ilm_image, self.get_slit_angle())
            # Apply undistortion
            # TODO check undistortion coeffs
            undistortion_coefficients = self.get_undistortion_coefficients()
            self.ilm_image = hsiutils.undistort_image(self.ilm_image, undistortion_coefficients)
            if len(self.ilm_image.shape) == 2:
                self.ilm_image_preview = cv.cvtColor(self.ilm_image, cv.COLOR_GRAY2RGB)
            else:
                self.ilm_image_preview = copy.deepcopy(self.ilm_image)
            image_to_draw_qt = self.image_to_qimage(self.ilm_image_preview)
            # QImage references ndarray data, so we need to copy QImage
            # See: https://stackoverflow.com/a/49516303
            self.send_ilm_image.emit(image_to_draw_qt.copy())

    @pyqtSlot(QRectF)
    def on_compute_slit_angle(self, area_rect: QRectF):
        x, y, w, h = area_rect.topLeft().x(), area_rect.topLeft().y(), area_rect.width(), area_rect.height()
        self.slit_data.slope, self.slit_data.angle, self.slit_data.intercept = \
            hsiutils.compute_slit_angle(self.slit_image, int(x), int(y), int(w), int(h),
                                        self.threshold_value, self.threshold_type)
        self.slit_data.slit_roi_origin_x = int(x)
        self.slit_data.slit_roi_origin_y = int(y)
        self.slit_data.slit_roi_width = int(w)
        self.slit_data.slit_roi_height = int(h)

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
        factors = np.array(self.slit_data.barrel_distortion_params['factors'])
        coeffs = np.array(self.slit_data.barrel_distortion_params['coeffs'])
        powers = np.array(self.slit_data.barrel_distortion_params['powers'])
        center_xy = np.array(self.slit_data.barrel_distortion_params['center'])
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
        factors = np.array(self.slit_data.barrel_distortion_params['factors'])
        coeffs = np.array(self.slit_data.barrel_distortion_params['coeffs'])
        powers = np.array(self.slit_data.barrel_distortion_params['powers'])
        center_xy = np.array(self.slit_data.barrel_distortion_params['center'])
        image = self.slit_image_rotated_rgb
        if contrasted:
            image = self.bd_slit_contrasted
        undistortion_coeffs = \
            list(hsiutils.compute_undistortion_coeffs(coeffs, powers, factors, image.shape, center_xy))
        self.slit_data.barrel_distortion_params['undistortion_coeffs'] = undistortion_coeffs
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
        self.slit_data.barrel_distortion_params['center'] = [center_x, center_y]
        self.send_bd_slit_center.emit(center_x, center_y)

    @pyqtSlot(bool)
    def on_apply_roi_ilm_image(self, apply: bool):
        if apply:
            self.ilm_image_roi = self.apply_roi(self.ilm_image)
            # self.ilm_image_roi_preview = cv.cvtColor(self.ilm_image_roi, cv.COLOR_GRAY2RGB)
            self.ilm_image_roi_preview = self.apply_roi(self.ilm_image_preview)
            image_qt = self.image_to_qimage(self.ilm_image_roi_preview)
            self.send_ilm_image.emit(image_qt.copy())
        else:
            image_qt = self.image_to_qimage(self.ilm_image_preview)
            self.send_ilm_image.emit(image_qt.copy())
