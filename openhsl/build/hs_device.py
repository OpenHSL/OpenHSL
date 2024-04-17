import numpy as np
from openhsl.build.hs_image_utils import BaseIntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Union
import openhsl.build.utils as utils


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
        self.x: int = 0
        self.y: int = 0
        self.width: int = 0
        self.height: int = 0

    def load_dict(self, data_dict: dict):
        if utils.key_exists_in_dict(data_dict, "slope"):
            self.slope = data_dict["slope"]
        if utils.key_exists_in_dict(data_dict, "angle"):
            self.angle = data_dict["angle"]
        if utils.key_exists_in_dict(data_dict, "intercept"):
            self.intercept = data_dict["intercept"]
        if utils.key_exists_in_dict(data_dict, "x"):
            self.x = data_dict["x"]
        if utils.key_exists_in_dict(data_dict, "y"):
            self.y = data_dict["y"]
        if utils.key_exists_in_dict(data_dict, "width"):
            self.width = data_dict["width"]
        if utils.key_exists_in_dict(data_dict, "height"):
            self.height = data_dict["height"]

    @classmethod
    def from_dict(cls, data_dict: dict):
        obj = cls()
        if utils.key_exists_in_dict(data_dict, "slope"):
            obj.slope = data_dict["slope"]
        if utils.key_exists_in_dict(data_dict, "angle"):
            obj.angle = data_dict["angle"]
        if utils.key_exists_in_dict(data_dict, "intercept"):
            obj.intercept = data_dict["intercept"]
        if utils.key_exists_in_dict(data_dict, "x"):
            obj.x = data_dict["x"]
        if utils.key_exists_in_dict(data_dict, "y"):
            obj.y = data_dict["y"]
        if utils.key_exists_in_dict(data_dict, "width"):
            obj.width = data_dict["width"]
        if utils.key_exists_in_dict(data_dict, "height"):
            obj.height = data_dict["height"]

        return obj

    def to_dict(self) -> dict:
        data = dict()
        data["slope"] = self.slope
        data["angle"] = self.angle
        data["intercept"] = self.intercept
        data["x"] = self.x
        data["y"] = self.y
        data["width"] = self.width
        data["height"] = self.height

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

    def get_slit_slope(self) -> float:
        return self.calib_slit_data.slope

    def get_slit_angle(self) -> float:
        return self.calib_slit_data.angle

    def get_slit_intercept(self, to_int = False) -> Union[int, float]:
        if to_int:
            return int(np.rint(self.calib_slit_data.intercept))
        return self.calib_slit_data.intercept

    def get_slit_roi(self) -> Tuple[int, int, int, int]:
        return self.calib_slit_data.x, self.calib_slit_data.y, self.calib_slit_data.width, self.calib_slit_data.height

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
