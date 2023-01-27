import numpy as np
from hsi import HSImage
from hs_raw_pb_data import RawCsvData, RawData
from Gaidel_Legacy.build import build_hypercube_by_videos


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

        device_type : str
            'uav' -
            'rotary' -
            'rail' -
            'gaidel_uav' -

        Attributes
        ----------
        hsi :

        Examples
        --------

    """

    def __init__(self, path_to_data, path_to_metadata=None, data_type=None):
        """

        """
        self.path_to_data = path_to_data
        self.path_to_metadata = path_to_metadata
        self.hsi = None
        self.frame_iterator = RawData(path_to_data=path_to_data, type_data=data_type)

        # if path_to_metadata:
        #     self.telemetry_iterator = RawCsvData(path_to_metadata, path_to_data)

    # ------------------------------------------------------------------------------------------------------------------

    # TODO must be realised
    def norm_frame_camera_illumination(self, frame: np.ndarray) -> np.ndarray:
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
    def norm_frame_camera_geometry(self, frame: np.ndarray) -> np.ndarray:
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

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """

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

    def geometry_normalization(self, frame: np.ndarray) -> np.ndarray:

        def rotate_frame(frame: np.ndarray) -> np.ndarray:
            """

            Parameters
            ----------
            self
            frame

            Returns
            -------

            """
            return frame
        # --------------------------------------------------------------------------------------------------------------

        def trapezoid_normalization(frame: np.ndarray) -> np.ndarray:
            """

            Parameters
            ----------
            frame

            Returns
            -------

            """
            return frame
        # --------------------------------------------------------------------------------------------------------------

        def barrel_normalization(frame: np.ndarray) -> np.ndarray:
            """

            Parameters
            ----------
            frame

            Returns
            -------

            """
            return frame
        # --------------------------------------------------------------------------------------------------------------

        frame = barrel_normalization(frame=frame)
        frame = trapezoid_normalization(frame=frame)
        frame = rotate_frame(frame=frame)

        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def _some_preparation_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        frame :

        Returns
        -------

        """
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def build(self, device_type: str, roi=False, gaidel=False):
        """
            Creates HSI from device-data
            # TODO must be realised from load_from_rail_dev, load_from_uav_dev and load_from_rot
        """
        if gaidel:
            data = build_hypercube_by_videos(self.path_to_data, "", self.path_to_metadata, "")
            data = np.transpose(data, (1, 2, 0))
            self.hsi = HSImage(hsi=data, wavelengths=None)
        else:
            preproc_frames = []

            for frame in self.frame_iterator:
                frame = self.norm_frame_camera_illumination(frame=frame)
                frame = self.norm_frame_camera_geometry(frame=frame)
                if roi:
                    frame = self.get_roi(frame)
                preproc_frames.append(frame)

            data = np.array(preproc_frames)
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
