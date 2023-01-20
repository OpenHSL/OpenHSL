import numpy as np

from hsi import HSImage
from hs_raw_pb_data import RawImagesData


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

    def __init__(self, hs_raw_data, metadata):
        self.hsi = None
        self.hs_raw_data = hs_raw_data
        self.metadata = metadata
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

    def load_from_rail_dev(self, rail_iterator):
        """
        Creates HSI from rail-device

        """
        hsi_tmp = []
        for frame in rail_iterator:
            tmp_layer = self._some_preparation_on_frame(frame=frame)
            hsi_tmp.append(tmp_layer)
        self.hsi = np.array(hsi_tmp).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_uav_dev(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_rot_dev(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi)
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------