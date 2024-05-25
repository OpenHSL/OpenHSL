import numpy as np
from openhsl.build.hs_image_utils import BaseIntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import openhsl.build.utils as utils
import openhsl.build.hs_image_utils as hsiutils


class HSCalibrationWavelengthData:
    def __init__(self):
        self.wavelength: Optional[int] = None
        self.calib_slit_slope: Optional[float] = None
        self.calib_slit_angle: Optional[float] = None
        self.calib_slit_intercept: Optional[float] = None
        self.wavelength_x: Optional[int] = None
        self.wavelength_y: Optional[int] = None
        self.wavelength_slit_offset: Optional[int] = None

    @classmethod
    def from_dict(cls, data_dict: dict):
        obj = cls()
        if utils.key_exists_in_dict(data_dict, "wavelength"):
            obj.wavelength = data_dict["wavelength"]
        if utils.key_exists_in_dict(data_dict, "calib_slit_slope"):
            obj.calib_slit_slope = data_dict["calib_slit_slope"]
        if utils.key_exists_in_dict(data_dict, "calib_slit_angle"):
            obj.calib_slit_angle = data_dict["calib_slit_angle"]
        if utils.key_exists_in_dict(data_dict, "calib_slit_intercept"):
            obj.calib_slit_intercept = data_dict["calib_slit_intercept"]
        if utils.key_exists_in_dict(data_dict, "wavelength_x"):
            obj.wavelength_x = data_dict["wavelength_x"]
        if utils.key_exists_in_dict(data_dict, "wavelength_y"):
            obj.wavelength_y = data_dict["wavelength_y"]
        if utils.key_exists_in_dict(data_dict, "wavelength_slit_offset"):
            obj.wavelength_slit_offset = data_dict["wavelength_slit_offset"]

        return obj

    def to_dict(self):
        data = dict()
        data["wavelength"] = self.wavelength
        data["calib_slit_slope"] = self.calib_slit_slope
        data["calib_slit_angle"] = self.calib_slit_angle
        data["calib_slit_intercept"] = self.calib_slit_intercept
        data["wavelength_x"] = self.wavelength_x
        data["wavelength_y"] = self.wavelength_y
        data["wavelength_slit_offset"] = self.wavelength_slit_offset

        return data


class HSDeviceType(BaseIntEnum):
    Undef = 0
    Rail = 1
    Rotor = 2
    UAV = 3
    UAV_Gaidel = 4
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
        self.calib_wavelength_data: Optional[List[HSCalibrationWavelengthData]] = None
        # ROI for slit
        self.calib_slit_data: Optional[HSCalibrationSlitData] = None

    def get_barrel_distortion_params(self) -> Dict[str, Union[List[float], Tuple[int]]]:
        return self.calib_slit_data.barrel_distortion_params

    def get_image_shape(self):
        return self.calib_slit_data.image_shape

    def get_slit_slope(self) -> float:
        return self.calib_slit_data.slope

    def get_slit_angle(self) -> float:
        return self.calib_slit_data.angle

    def get_slit_intercept(self, to_int=False) -> Union[int, float]:
        if to_int:
            return int(np.rint(self.calib_slit_data.intercept))
        return self.calib_slit_data.intercept

    def get_slit_roi(self) -> Tuple[int, int, int, int]:
        return self.calib_slit_data.slit_roi_origin_x, self.calib_slit_data.slit_roi_origin_y, \
            self.calib_slit_data.slit_roi_width, self.calib_slit_data.slit_roi_height

    def get_undistortion_coefficients(self) -> List[float]:
        undistortion_coeffs = None
        compute_needed = False
        if utils.key_exists_in_dict(self.get_barrel_distortion_params(), 'undistortion_coeffs'):
            undistortion_coeffs = self.calib_slit_data.barrel_distortion_params['undistortion_coeffs']
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
            self.calib_slit_data.barrel_distortion_params['undistortion_coeffs'] = undistortion_coeffs
        return undistortion_coeffs


    def load_calibration_wavelength_data(self, path: Union[str, Path]) -> None:
        pass

    def load_device_data(self, path: Union[str, Path]) -> None:
        pass

    def load_dict(self, device_data: dict):
        if utils.key_exists_in_dict(device_data, "device_type"):
            self.device_type = device_data["device_type"]
        if utils.key_exists_in_dict(device_data, "calib_wavelength_data"):
            calib_wavelength_data_dict = device_data["calib_wavelength_data"]
            calib_wavelength_data = \
                [HSCalibrationWavelengthData.from_dict(v) for v in calib_wavelength_data_dict.values()]
            self.calib_wavelength_data = calib_wavelength_data
        if utils.key_exists_in_dict(device_data, "calib_slit_data"):
            self.calib_slit_data = HSCalibrationSlitData.from_dict(device_data["calib_slit_data"])

    @classmethod
    def from_dict(cls, device_data: dict):
        obj = cls()
        if utils.key_exists_in_dict(device_data, "device_type"):
            obj.device_type = device_data["device_type"]
        if utils.key_exists_in_dict(device_data, "calib_wavelength_data"):
            calib_wavelength_data_dict = device_data["calib_wavelength_data"]
            calib_wavelength_data = \
                [HSCalibrationWavelengthData.from_dict(v) for v in calib_wavelength_data_dict.values()]
            obj.calib_wavelength_data = calib_wavelength_data
        if utils.key_exists_in_dict(device_data, "calib_slit_data"):
            obj.calib_slit_data = HSCalibrationSlitData.from_dict(device_data["calib_slit_data"])

        return obj

    def to_dict(self):
        device_data = dict()
        device_data["device_type"] = self.device_type

        if self.calib_wavelength_data is None:
            device_data["calib_wavelength_data"] = None
        else:
            calib_wavelength_data = dict()
            for wl in self.calib_wavelength_data:
                calib_wavelength_data[str(wl.wavelength)] = wl.to_dict()
            device_data["calib_wavelength_data"] = calib_wavelength_data

        if self.calib_slit_data is None:
            device_data["calib_slit_data"] = None
        else:
            device_data["calib_slit_data"] = self.calib_slit_data.to_dict()

        return device_data
