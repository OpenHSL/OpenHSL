from typing import Optional
from hs_mask import HSMask
from hsi import HSImage


class Dataloader:
    """
    Dataloader()

        Creates loader from HSImage and HSMask (optional)

        Has built-in Sampler

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

    def __init__(self, hsi: Optional[HSImage], mask: Optional[HSMask]):
        self.hsi = hsi.data
        self.mask = mask.data
