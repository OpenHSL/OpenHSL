from hsi import HSImage
from hs_raw_pushbroom_data import HSRawData

class HSBuilder:
    """
    Build a HSI object from HSRawData

    """

    def __init__(self, hs_raw_data: HSRawData, metadata_):
        pass


    def some_preparation_on_hsi(self):
        pass

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi)
        except:
            pass
