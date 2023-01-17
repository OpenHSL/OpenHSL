from numpy import array

from Mask import Mask
from HSI import HSImage

class Dataloader:

    hsi: array
    mask : array

    def __init__(self, hsi: HSImage, mask: Mask):
        self.hsi = hsi.data
        self.mask = mask.data
