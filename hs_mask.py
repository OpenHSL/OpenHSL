from typing import Optional, Dict
import numpy as np


class HSMask:
    """
    HSMask()

        Image-like object which contains:
            2D-Array
                Each pixel has value from [0, class_counts - 1]
            3D-Array
                Each layer is binary image where 1 is class and 0 is not-class

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

    def __init__(self, mask: Optional[np.array], metadata: Optional[Dict]):
        self.data = mask
        pass

    def load_from_png(self):
        pass

    def load_from_bmp(self):
        pass

    def load_from_mat(self):
        pass

    def load_from_TEMPLATE(self):
        pass
