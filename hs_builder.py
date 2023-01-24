import numpy as np

from hsi import HSImage
from hs_raw_pb_data import RawImagesData, RawVideoData, RawCsvData


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

        Attributes
        ----------
        hsi :

        Examples
        --------

    """

    def __init__(self, path_to_data, path_to_metadata=None, data_type=None, device_type=None):
        """

        """
        self.hsi = None

        if data_type == 'images':
            self.frame_iterator = RawImagesData(path_to_data)
        elif data_type == 'video':
            self.frame_iterator = RawVideoData(path_to_data)
        else:
            # TODO other cases
            pass

        if path_to_metadata:
            self.telemetry_iterator = RawCsvData(path_to_metadata)

        if device_type == 'uav':
            self.load_from_uav_dev()
        elif device_type == 'rotary':
            self.load_from_rot_dev()
        elif device_type == 'rail':
            self.load_from_rail_dev()
        else:
            # TODO other cases
            pass
    # ------------------------------------------------------------------------------------------------------------------

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        frame :

        Returns
        -------

        """
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def illumination_normalization(self, frame: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        frame

        Returns
        -------

        """
        return frame
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

    def rotary_normalization(self, frame: np.ndarray, grad: float) -> np.ndarray:
        """

        Parameters
        ----------
        frame
        grad

        Returns
        -------

        """
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

    def load_from_rail_dev(self):
        """
            Creates HSI from rail-device
            Steps:
                1) Takes target area from raw-frame
                2) Correction barrel distortion
                3) Correction trapezoid distortion
                4) Correction illuminate heterogeneity
                5) Future corrections
                6) Added to Y-Z layers set
        """
        hsi_tmp = []
        for frame in self.frame_iterator:
            tmp_layer = self.illumination_normalization(frame=frame)
            tmp_layer = self.geometry_normalization(frame=tmp_layer)
            tmp_layer = self._some_preparation_on_frame(frame=tmp_layer)
            tmp_layer = self.get_roi(frame=tmp_layer)
            hsi_tmp.append(np.array(tmp_layer))

        data = np.array(hsi_tmp)
        self.hsi = HSImage(hsi=data, metadata=None)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_uav_dev(self):
        """
            Creates HSI from uav-device
            #TODO REPLACE to here spectru CODE!
        """
        # Oh shit. here we go again
        # TODO get min-max coordinates
        # TODO Create empy hsi by these coordinates
        for frame, telem in zip(self.frame_iterator, self.telemetry_iterator):
            # TODO get coordinates of the line ends
            # TODO add to hsi this line
            ...
        # TODO interpolate empty spaces between lines
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_rot_dev(self):
        """
            Creates HSI from rotary-device
            Steps:
                1) Takes target area from raw-frame
                2) Correction barrel distortion
                3) Correction trapezoid distortion
                4) Correction illuminate heterogeneity
                5) Correction rotary distortion
                6) Future corrections
                7) Added to Y-Z layers set
        """
        hsi_tmp = []
        for frame, telem in (self.frame_iterator, self.telemetry_iterator):
            tmp_layer = self.geometry_normalization(frame=frame)
            tmp_layer = self.illumination_normalization(frame=tmp_layer)
            tmp_layer = self.rotary_normalization(frame=tmp_layer, grad=telem)
            tmp_layer = self._some_preparation_on_frame(frame=tmp_layer)
            tmp_layer = self.get_roi(frame=tmp_layer)
            hsi_tmp.append(tmp_layer)

        data = np.array(hsi_tmp)
        self.hsi = HSImage(hsi=data, metadata=None)
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        """

        Returns
        -------

        """
        try:
            return self.hsi
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------
