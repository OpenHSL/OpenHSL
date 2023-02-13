import numpy as np
from hsi import HSImage
from hs_raw_pb_data import RawCsvData, RawData
from gaidel_legacy import build_hypercube_by_videos
from typing import Optional
import settings
from utils import gaussian


class HSBuilder:
    """
    HSBuilder(path_to_data, path_to_metadata=None, data_type=None)

        Build a HSI object from HSRawData

        Parameters
        ----------
        path_to_data : str

        path_to_metadata : str

        data_type : str
            'images' -
            'video' -

        Attributes
        ----------
        hsi : HSImage
        frame_iterator: RawData
        telemetry_iterator: RawCsvData
        Examples
        --------

    """

    def __init__(self, path_to_data, path_to_metadata=None, data_type=None):
        """

        """
        self.path_to_data = path_to_data
        self.path_to_metadata = path_to_metadata
        self.hsi: Optional[HSImage] = None
        self.frame_iterator = RawData(path_to_data=path_to_data, type_data=data_type)

        # if path_to_metadata != None:
        #     self.telemetry_iterator = RawCsvData(path_to_metadata, path_to_data)

    # ------------------------------------------------------------------------------------------------------------------

    # TODO must be realised
    def __norm_frame_camera_illumination(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalizes illumination on frame.
        Frame have heterogeneous illumination by slit defects. It must be solved

        Parameters
        ----------
        frame

        Returns
        -------
        frame
        """
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    # TODO must be realised
    def __norm_frame_camera_geometry(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalizes geometric distortions on frame.

        Parameters
        ----------
        frame

        Returns
        -------
        frame

        """
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    # TODO must be realised
    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        For this moment works to microscope rough settings
        Parameters
        ----------
        frame :

        Returns
        -------

        """
        gap_coord = 620
        range_to_spectrum = 185
        range_to_end_spectrum = 250
        left_bound_spectrum = 490
        right_bound_spectrum = 1390
        x1 = gap_coord + range_to_spectrum
        x2 = x1 + range_to_end_spectrum
        return frame[x1: x2, left_bound_spectrum: right_bound_spectrum].T
    # ------------------------------------------------------------------------------------------------------------------

    # TODO must be realised
    def __some_preparation_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        frame :

        Returns
        -------

        """
        return frame
# ------------------------------------------------------------------------------------------------------------------

    def __principal_slices(self, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame

        Returns
        -------

        """
        width, height = frame.shape # 744, 70
        gaus_width = height // settings.BANDS_NUMBERS
        gaussian_window = gaussian(gaus_width, gaus_width / 2.0, gaus_width / 6.0)
        mid = len(gaussian_window) // 2
        gaussian_window[mid] = 1.0 - gaussian_window[:mid].sum() - gaussian_window[mid+1:].sum()
        result = np.zeros((width, settings.BANDS_NUMBERS), dtype=np.uint8)
        for i in range(settings.BANDS_NUMBERS):
            result[:, i] = np.sum(frame[:, i * gaus_width:(i + 1) * gaus_width] * gaussian_window, axis=1)

        return result
# ------------------------------------------------------------------------------------------------------------------

    def build(self, roi=False):
        """
            Creates HSI from device-data
        """
        preproc_frames = []
        for frame in self.frame_iterator:
            frame = self.__norm_frame_camera_illumination(frame=frame)
            frame = self.__norm_frame_camera_geometry(frame=frame)
            frame = self.__some_preparation_on_frame(frame=frame)
            frame = self.__principal_slices(frame = frame)
            if roi:
                frame = self.get_roi(frame)
            preproc_frames.append(frame)
            
        data = np.array(preproc_frames)
        if self.path_to_metadata:
            data = build_hypercube_by_videos(data, self.path_to_metadata)
            
        self.hsi = HSImage(hsi=data, wavelengths=None)
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        """

        Returns
        -------
        self.hsi : HSImage
            Builded from source hsi object

        """
        try:
            return self.hsi
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------
