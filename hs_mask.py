import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from typing import Optional, Dict
import os.path


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
        mask: np.ndarray
            3D-matrix which has a dimension X - Y - Z.
            where:
                X, Y data resolution.
                Z is a count of channels (1, 3, 4).
        Attributes
        ----------
        data: np.ndarray

        metadata: dict

        Examples
        --------
            arr = np.zeros((100, 100, 3))
            md = {'1':'class_1', '2':'class_2'}

            hsi = HSMask(hsi=arr, metadata=md)
    """

    def __init__(self, mask: Optional[np.array], label_class: Optional[Dict]):
        self.data = mask
        self.label_class = label_class
        self.n_classes = len(np.unique(mask))
        
    # ------------------------------------------------------------------------------------------------------------------
    def load_mask(self, path_to_file: str, mat_key: str = None,  h5_key: str = None):

        """
        load_mask(path_to_file, mat_key, h5_key)

            Reads information from a file,
            converting it to the numpy.array format

            input data shape:
            ____________
            3-dimensional images in png, bmp, jpg
            format or h5, math, npy files are submitted to the input
            ____________

            Parameters
            ----------
            path_to_file: str
                Path to file
            mat_key: str
                Key for field in .mat file as dict object
                mat_file['image']
            h5_key: str
                Key for field in .h5 file as 5h object

        """
        def load_img(path_to_file):
            """
            ____________
            necessary for reading 3-dimensional images
            ____________

            Parameters
            ----------
            path_to_file: str
                Path to file
            """
            hsi = []
            img = Image.open(path_to_file).convert("L")
            hsi.append(np.array(img))
            return np.array(hsi).transpose((1, 2, 0))

        _, file_extension = os.path.splitext(path_to_file)

        if file_extension in ['.jpg','.jpeg','.bmp','.png']:
            '''
            loading a mask from images
            '''
            self.data = load_img(path_to_file)

        elif file_extension == '.npy':
            '''
            loading a mask from numpy file
            '''
            self.data = np.load(path_to_file)

        elif file_extension == '.mat':
            '''
            loading a mask from mat file
            '''
            self.data = loadmat(path_to_file)[mat_key]

        elif file_extension == '.h5':
            '''
            loading a mask from h5 file
            '''
            self.data = h5py.File(path_to_file, 'r')[h5_key]

        # updates number of classes after loading mask
        self.n_classes = len(np.unique(self.data))
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_mat(self, path_to_file: str, mat_key: str):
        """
        save_to_mat(path_to_file, mat_key)

            ____________
            save the mask in mat format
            ____________

            Parameters
            ----------
            path_to_file: str
                Path to file
            mat_key: str
                Key for field in .mat file as dict object
                mat_file['image']
        """
        temp_dict = {mat_key: self.data}
        savemat(path_to_file, temp_dict)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_h5(self, path_to_file: str, h5_key: str):
        """
        save_to_h5(path_to_file, h5_key)

        ____________
        save the mask in h5 format
        ____________

        Parameters
        ----------
        path_to_file: str
            Path to file
        mat_key: str
            Key for field in .mat file as dict object
            mat_file['image']
        h5_key: str
            Key for field in .h5 file as 5h object
        """

        with h5py.File(path_to_file, 'w') as f:
            f.create_dataset(h5_key, data=self.data)
    # ------------------------------------------------------------------------------------------------------------------
    
    def save_to_npy(self, path_to_file: str):
        '''
        save_to_npy(path_to_file)

        ____________
        save the mask in numpy format
        ____________

        Parameters
        ----------
        path_to_file: str
            Path to file
        '''
        np.save(path_to_file, self.data)
    # ------------------------------------------------------------------------------------------------------------------

    def save_image(self, path_to_save_file: str):
        '''
        save_image(path_to_save_file)

        ____________
        save the mask in 'jpg','jpeg','bmp','png' format
        ____________

        Parameters
        ----------
        path_to_file: str
            Path to file
        '''
        img = Image.fromarray(self.data[:,:,0])
        img.save(path_to_save_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_class_info(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def save_class_info(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------


