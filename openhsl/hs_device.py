from openhsl.hs_image_utils import BaseIntEnum
from pathlib import Path
from typing import List, Optional, Union
import openhsl.utils as utils


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


class HSROI:
    def __init__(self):
        self.slit_slope: float = 0
        self.slit_angle: float = 0
        self.slit_intercept: int = 0
        self.x: int = 0
        self.y: int = 0
        self.width: int = 0
        self.height: int = 0

    @classmethod
    def from_dict(cls, data_dict: dict):
        obj = cls()
        if utils.key_exists_in_dict(data_dict, "slit_slope"):
            obj.slit_slope = data_dict["slit_slope"]
        if utils.key_exists_in_dict(data_dict, "slit_angle"):
            obj.slit_angle = data_dict["slit_angle"]
        if utils.key_exists_in_dict(data_dict, "slit_intercept"):
            obj.slit_intercept = data_dict["slit_intercept"]
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
        data["slit_slope"] = self.slit_slope
        data["slit_angle"] = self.slit_angle
        data["slit_intercept"] = self.slit_intercept
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
        self.roi: Optional[HSROI] = None

    def load_calibration_wavelength_data(self, path: Union[str, Path]) -> None:
        pass

    def load_device_data(self, path: Union[str, Path]) -> None:
        pass

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
        if utils.key_exists_in_dict(device_data, "roi"):
            obj.roi = HSROI.from_dict(device_data["roi"])

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

        if self.roi is None:
            device_data["roi"] = None
        else:
            device_data["roi"] = self.roi.to_dict()

        return device_data
