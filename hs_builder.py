from hsi import HSImage
from hs_raw_pushbroom_data import HSRawData

class HSBuilder:
    """
    HSBuilder()

        Build a HSI object from HSRawData

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

    def __init__(self, hs_raw_data: HSRawData, metadata_):
        pass

    def load_from_rail_dev(self):
        pass

    def load_from_uav_dev(self):
        pass

    def load_from_rot_dev(self):
        pass

    def some_preparation_on_hsi(self):
        pass

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi)
        except:
            pass
