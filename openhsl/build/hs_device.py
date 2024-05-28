import copy

import numpy as np
from openhsl.build.hs_image_utils import BaseIntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import openhsl.build.utils as utils
import openhsl.build.hs_image_utils as hsiutils


class HSCalibrationWavelengthData:
    def __init__(self):
        self.wavelength_list: Optional[List[float]] = None
        self.wavelength_y_list: Optional[List[int]] = None
        self.wavelength_slit_offset_y_list: Optional[List[int]] = None
        self.spectrum_slit_offset_y: int = 0
        self.spectrum_left_bound: int = 0
        self.spectrum_right_bound: int = 0

    @classmethod
    def from_dict(cls, data_dict: dict):
        obj = cls()
        if utils.key_exists_in_dict(data_dict, "wavelength_list"):
            obj.wavelength_list = data_dict["wavelength_list"]
        if utils.key_exists_in_dict(data_dict, "wavelength_y_list"):
            obj.wavelength_y_list = data_dict["wavelength_y_list"]
        if utils.key_exists_in_dict(data_dict, "wavelength_slit_offset_y_list"):
            obj.wavelength_slit_offset_y_list = data_dict["wavelength_slit_offset_y_list"]
        if utils.key_exists_in_dict(data_dict, "spectrum_slit_offset_y"):
            obj.spectrum_slit_offset_y = data_dict["spectrum_slit_offset_y"]
        if utils.key_exists_in_dict(data_dict, "spectrum_left_bound"):
            obj.spectrum_left_bound = data_dict["spectrum_left_bound"]
        if utils.key_exists_in_dict(data_dict, "spectrum_right_bound"):
            obj.spectrum_right_bound = data_dict["spectrum_right_bound"]

        return obj

    def to_dict(self):
        data = dict()
        data["wavelength_list"] = self.wavelength_list
        data["wavelength_y_list"] = self.wavelength_y_list
        data["wavelength_slit_offset_y_list"] = self.wavelength_slit_offset_y_list
        data["spectrum_slit_offset_y"] = self.spectrum_slit_offset_y
        data["spectrum_left_bound"] = self.spectrum_left_bound
        data["spectrum_right_bound"] = self.spectrum_right_bound

        return data


class HSDeviceType(BaseIntEnum):
    Undef = 0
    Rail = 1
    Rotor = 2
    UAV = 3
    Custom = 256


class HSCalibrationSlitData:
    def __init__(self):
        self.slope: float = 0
        self.angle: float = 0
        self.intercept: float = 0
        self.slit_roi_origin_x: int = 0
        self.slit_roi_origin_y: int = 0
        self.slit_roi_width: int = 0
        self.slit_roi_height: int = 0
        self.image_shape: Optional[Tuple[int, ...]] = None
        # {'center': [], 'powers': [], 'coeffs': [], 'factors': []}
        self.barrel_distortion_params: Optional[Dict[str, Union[List[float]], List[int]]] = None

    def load_dict(self, data_dict: dict):
        if utils.key_exists_in_dict(data_dict, "slope"):
            self.slope = data_dict["slope"]
        if utils.key_exists_in_dict(data_dict, "angle"):
            self.angle = data_dict["angle"]
        if utils.key_exists_in_dict(data_dict, "intercept"):
            self.intercept = data_dict["intercept"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_origin_x"):
            self.slit_roi_origin_x = data_dict["slit_roi_origin_x"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_origin_y"):
            self.slit_roi_origin_y = data_dict["slit_roi_origin_y"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_width"):
            self.slit_roi_width = data_dict["slit_roi_width"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_height"):
            self.slit_roi_height = data_dict["slit_roi_height"]
        if utils.key_exists_in_dict(data_dict, "image_shape"):
            self.image_shape = tuple(data_dict["image_shape"])
        if utils.key_exists_in_dict(data_dict, "barrel_distortion_params"):
            self.barrel_distortion_params = data_dict["barrel_distortion_params"]

    @classmethod
    def from_dict(cls, data_dict: dict):
        obj = cls()
        if utils.key_exists_in_dict(data_dict, "slope"):
            obj.slope = data_dict["slope"]
        if utils.key_exists_in_dict(data_dict, "angle"):
            obj.angle = data_dict["angle"]
        if utils.key_exists_in_dict(data_dict, "intercept"):
            obj.intercept = data_dict["intercept"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_origin_x"):
            obj.slit_roi_origin_x = data_dict["slit_roi_origin_x"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_origin_y"):
            obj.slit_roi_origin_y = data_dict["slit_roi_origin_y"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_width"):
            obj.slit_roi_width = data_dict["slit_roi_width"]
        if utils.key_exists_in_dict(data_dict, "slit_roi_height"):
            obj.slit_roi_height = data_dict["slit_roi_height"]
        if utils.key_exists_in_dict(data_dict, "image_shape"):
            obj.image_shape = tuple(data_dict["image_shape"])
        if utils.key_exists_in_dict(data_dict, "barrel_distortion_params"):
            obj.barrel_distortion_params = data_dict["barrel_distortion_params"]

        return obj

    def to_dict(self) -> dict:
        data = dict()
        data["slope"] = self.slope
        data["angle"] = self.angle
        data["intercept"] = self.intercept
        data["slit_roi_origin_x"] = self.slit_roi_origin_x
        data["slit_roi_origin_y"] = self.slit_roi_origin_y
        data["slit_roi_width"] = self.slit_roi_width
        data["slit_roi_height"] = self.slit_roi_height
        data["image_shape"] = self.image_shape
        data["barrel_distortion_params"] = self.barrel_distortion_params

        return data


class HSDevice:
    """
    HSDevice()

        Preprocess HS frames from camera

        Parameters
        ----------

        Attributes
        ----------

        See Also
        --------

        Notes
        -----

        Examples
        --------

    """

    def __init__(self):
        self.device_type: HSDeviceType = HSDeviceType.Undef
        # ROI for slit
        self.slit_data: Optional[HSCalibrationSlitData] = None
        self.wavelength_data: Optional[HSCalibrationWavelengthData] = None
        self.illumination_mask: Optional[np.ndarray] = None

    def apply_roi(self, frame: np.ndarray):
        spectrum_left_bound = self.wavelength_data.spectrum_left_bound
        spectrum_right_bound = self.wavelength_data.spectrum_right_bound
        idx_y = self.wavelength_data.wavelength_slit_offset_y_list
        idx_y = list(map(lambda x: x + self.wavelength_data.spectrum_slit_offset_y, idx_y))
        idx_x = list(range(spectrum_left_bound, spectrum_right_bound + 1))

        frame_roi: Optional[np.ndarray] = None
        if len(frame.shape) == 2:
            ixgrid = np.ix_(idx_y, idx_x)
            frame_roi = copy.deepcopy(frame[ixgrid[0], ixgrid[1]])
        else:
            ixgrid = np.ix_(idx_y, idx_x, [0, 1, 2])
            frame_roi = copy.deepcopy(frame[ixgrid[0], ixgrid[1], ixgrid[2]])
        return frame_roi

    def get_barrel_distortion_params(self) -> Dict[str, Union[List[float], Tuple[int]]]:
        return self.slit_data.barrel_distortion_params

    def get_image_shape(self):
        return self.slit_data.image_shape

    def get_slit_slope(self) -> float:
        return self.slit_data.slope

    def get_slit_angle(self) -> float:
        return self.slit_data.angle

    def get_slit_intercept(self, to_int=False) -> Union[int, float]:
        if to_int:
            return int(np.rint(self.slit_data.intercept))
        return self.slit_data.intercept

    def get_slit_roi(self) -> Tuple[int, int, int, int]:
        return self.slit_data.slit_roi_origin_x, self.slit_data.slit_roi_origin_y, \
            self.slit_data.slit_roi_width, self.slit_data.slit_roi_height

    def get_undistortion_coefficients(self) -> List[float]:
        undistortion_coeffs = None
        compute_needed = False
        if utils.key_exists_in_dict(self.get_barrel_distortion_params(), 'undistortion_coeffs'):
            undistortion_coeffs = self.slit_data.barrel_distortion_params['undistortion_coeffs']
        else:
            compute_needed = True
        if undistortion_coeffs is None:
            compute_needed = True
        elif len(undistortion_coeffs) == 0:
            compute_needed = False
        if compute_needed:
            barrel_distortion_params = self.get_barrel_distortion_params()
            coeffs = np.array(barrel_distortion_params['coeffs'])
            powers = np.array(barrel_distortion_params['powers'])
            factors = np.array(barrel_distortion_params['factors'])
            image_shape = self.get_image_shape()
            center_xy = np.array(barrel_distortion_params['center'])
            undistortion_coeffs = hsiutils.compute_undistortion_coeffs(coeffs, powers, factors, image_shape, center_xy)
            undistortion_coeffs = undistortion_coeffs.tolist()
            self.slit_data.barrel_distortion_params['undistortion_coeffs'] = undistortion_coeffs
        return undistortion_coeffs


    def load_calibration_wavelength_data(self, path: Union[str, Path]) -> None:
        pass

    def load_device_data(self, path: Union[str, Path]) -> None:
        pass

    def load_dict(self, device_data: dict):
        if utils.key_exists_in_dict(device_data, "device_type"):
            self.device_type = device_data["device_type"]
        if utils.key_exists_in_dict(device_data, "slit_data"):
            self.slit_data = HSCalibrationSlitData.from_dict(device_data["slit_data"])
        if utils.key_exists_in_dict(device_data, "wavelength_data"):
            self.wavelength_data = HSCalibrationWavelengthData.from_dict(device_data["wavelength_data"])

    @classmethod
    def from_dict(cls, device_data: dict):
        obj = cls()
        if utils.key_exists_in_dict(device_data, "device_type"):
            obj.device_type = device_data["device_type"]
        if utils.key_exists_in_dict(device_data, "slit_data"):
            obj.slit_data = HSCalibrationSlitData.from_dict(device_data["slit_data"])
        if utils.key_exists_in_dict(device_data, "wavelength_data"):
            obj.wavelength_data = HSCalibrationWavelengthData.from_dict(device_data["wavelength_data"])

        return obj

    def to_dict(self):
        device_data = dict()
        device_data["device_type"] = self.device_type

        if self.wavelength_data is None:
            device_data["wavelength_data"] = None
        else:
            device_data["wavelength_data"] = self.wavelength_data.to_dict()

        if self.slit_data is None:
            device_data["slit_data"] = None
        else:
            device_data["slit_data"] = self.slit_data.to_dict()

        return device_data
