from openhsl.hs_image_utils import BaseIntEnum
from pathlib import Path
from typing import List, Optional, Union


class HSCalibrationWavelengthData:
    def __init__(self):
        self.wavelength: Optional[int] = None
        self.calib_slit_slope: Optional[float] = None
        self.calib_slit_angle: Optional[float] = None
        self.calib_slit_intercept: Optional[float] = None
        self.wavelength_x: Optional[int] = None
        self.wavelength_y: Optional[int] = None
        self.wavelength_slit_offset: Optional[int] = None


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
