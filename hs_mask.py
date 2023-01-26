import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from typing import Optional, Dict


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
        self.metadata = metadata
        
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_png(self, path_to_file: str):

        self.data = np.array(Image.open(path_to_file))
        
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_bmp(self, path_to_file: str):

        self.data = np.array(Image.open(path_to_file))
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_mat(self, path_to_file: str, mat_key: str):

        self.data = loadmat(path_to_file)[mat_key]
    # ------------------------------------------------------------------------------------------------------------------
    def load_from_npy(self, path_to_file: str):

        self.data = np.load(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_h5(self, path_to_file: str, h5_key: str = None):

        self.data = h5py.File(path_to_file, 'r')[h5_key]
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_mat(self, path_to_file: str, mat_key: str):
  
        temp_dict = {mat_key: self.data}
        savemat(path_to_file, temp_dict)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_h5(self, path_to_file: str, h5_key: str):

        with h5py.File(path_to_file, 'w') as f:
            f.create_dataset(h5_key, data=self.data)
    # ------------------------------------------------------------------------------------------------------------------
    
    def save_to_npy(self, path_to_file: str):

        np.save(path_to_file, self.data)
    # ------------------------------------------------------------------------------------------------------------------

    
