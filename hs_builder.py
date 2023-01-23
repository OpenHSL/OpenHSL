import numpy as np

from hsi import HSImage
from hs_raw_pb_data import RawImagesData, RawVideoData


class HSBuilder:
    """
    HSBuilder()

        Build a HSI object from HSRawData

        Parameters
        ----------

        Attributes
        ----------

        Notes
        -----

        Examples
        --------

    """

    def __init__(self, path_to_data, metadata=None, data_type='images'):
        self.hsi = None
        self.path_to_data = path_to_data
        self.metadata = metadata
        self.data_type = data_type

        if data_type == 'images':
            self.frame_iterator = RawImagesData(self.path_to_data)
        elif data_type == 'video':
            self.frame_iterator = RawVideoData(self.path_to_data)
        else:
            # TODO other cases
            pass

        if metadata:
            self.telemetry = self.get_telem_from_metadata(metadata)
    # ------------------------------------------------------------------------------------------------------------------

    def get_telem_from_metadata(self, metadata):
        return ...

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def illumination_normalization(self, frame: np.ndarray) -> np.ndarray:
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def trapezoid_normalization(self, frame: np.ndarray) -> np.ndarray:
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def barrel_normalization(self, frame: np.ndarray) -> np.ndarray:
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def rotary_normalization(self, frame: np.ndarray, grad: float) -> np.ndarray:
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
            tmp_layer = self.barrel_normalization(frame=tmp_layer)
            tmp_layer = self.trapezoid_normalization(frame=tmp_layer)
            tmp_layer = self._some_preparation_on_frame(frame=tmp_layer)
            tmp_layer = self.get_roi(frame=tmp_layer)
            hsi_tmp.append(np.array(tmp_layer))

        # this transpose is needed for replace axis for z-x-y to x-y-z
        self.hsi = np.array(hsi_tmp).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_uav_dev(self, telem):
        """
            Creates HSI from uav-device
            #TODO REPLACE to here spectru CODE!
        """
        for frame, tm in zip(self.frame_iterator, telem):
            ...
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_rot_dev(self, telem):
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
        for frame, grad in (self.frame_iterator, telem):
            tmp_layer = self.get_roi(frame=frame)
            tmp_layer = self.barrel_normalization(frame=tmp_layer)
            tmp_layer = self.trapezoid_normalization(frame=tmp_layer)
            tmp_layer = self.illumination_normalization(frame=tmp_layer)
            tmp_layer = self.rotary_normalization(frame=tmp_layer, grad=grad)
            tmp_layer = self._some_preparation_on_frame(frame=tmp_layer)
            hsi_tmp.append(np.array(tmp_layer))

        # this transpose is needed for replace axis for z-x-y to x-y-z
        self.hsi = np.array(hsi_tmp).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi, metadata=None)
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------
