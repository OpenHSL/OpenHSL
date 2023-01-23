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

    def __init__(self, path_to_data, metadata):
        self.hsi = None
        self.path_to_data = path_to_data
        self.metadata = metadata
    # ------------------------------------------------------------------------------------------------------------------

    def cut_target_area(self, frame: np.ndarray) -> np.ndarray:
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def illumination_normalization(self, frame: np.ndarray) -> np.ndarray:
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def trapezoid_normalization(self, frame: np.ndarray) -> np.ndarray:
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def barrel_normalization(self, frame: np.ndarray) -> np.ndarray:
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def rotary_normalization(self, frame: np.ndarray) -> np.ndarray:
        pass
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _some_preparation_on_frame(frame: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        frame :

        Returns
        -------

        """
        return frame
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_rail_dev(self, rail_iterator):
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
        for frame in rail_iterator:
            tmp_layer = self.cut_target_area(frame=frame)
            tmp_layer = self.barrel_normalization(frame=tmp_layer)
            tmp_layer = self.trapezoid_normalization(frame=tmp_layer)
            tmp_layer = self.illumination_normalization(frame=tmp_layer)
            tmp_layer = HSBuilder._some_preparation_on_frame(frame=tmp_layer)
            hsi_tmp.append(np.array(tmp_layer))

        # this transpose is needed for replace axis for z-x-y to x-y-z
        self.hsi = np.array(hsi_tmp).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_uav_dev(self, uav_iterator):
        """
            Creates HSI from uav-device
            #TODO REPLACE to here spectru CODE!
        """
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_rot_dev(self, rotary_iterator):
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
        for frame in rotary_iterator:
            tmp_layer = self.cut_target_area(frame=frame)
            tmp_layer = self.barrel_normalization(frame=tmp_layer)
            tmp_layer = self.trapezoid_normalization(frame=tmp_layer)
            tmp_layer = self.illumination_normalization(frame=tmp_layer)
            tmp_layer = self.rotary_normalization(frame=tmp_layer)
            tmp_layer = HSBuilder._some_preparation_on_frame(frame=tmp_layer)
            hsi_tmp.append(np.array(tmp_layer))

        # this transpose is needed for replace axis for z-x-y to x-y-z
        self.hsi = np.array(hsi_tmp).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi)
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------