import h5py
import numpy as np
from os import listdir, mkdir
from os.path import isdir
from PIL import Image
from scipy.io import loadmat, savemat


class HSImage:
    """
    Hyperspectral Image which has a dimension X - Y - Z where Z is a count of channels
    ...
    Attributes
    ----------
    hsi : np.ndarray
        Hyperspectral Image in array format

    Methods
    -------


    """
    def __init__(self, hsi: np.ndarray = None):
        """
        Initializes HSI object

        Parameters
        ----------
        hsi: np.ndarray

        """
        self.hsi = hsi

    def load_from_mat(self, path_to_file: str, mat_key: str):
        """
        Loads HSI from .mat file

        Parameters
        ----------
        path_to_file: str
            Path to .mat file
        mat_key: str
            Key for field in .mat file as dict object
            mat_file['image']
        """
        self.hsi = loadmat(path_to_file)[mat_key]

    def load_from_tiff(self, path_to_file: str):
        """
        Loads HSI from .tiff file

        Parameters
        ----------
        path_to_file: str
            Path to .tiff file
        """
        ...

    def load_from_npy(self, path_to_file: str):
        """
        Loads HSI from .npy file

        Parameters
        ----------
        path_to_file: str
            Path to .npy file
        """
        self.hsi = np.load(path_to_file)

    def load_from_h5(self, path_to_file: str, h5_key: str = None):
        """
        Loads HSI from .h5 file

        Parameters
        ----------
        path_to_file: str
            Path to .h5 file
        h5_key: str
            Key for field in .h5 file as dict object
        """
        self.hsi = h5py.File(path_to_file, 'r')[h5_key]

    def load_from_images(self, path_to_dir: str):
        """
        Loads HSI from images are placed in directory

        Parameters
        ----------
        path_to_dir: str
            Path to directory with images
        """
        images_list = listdir(path_to_dir)
        hsi = []
        for image_name in images_list:
            img = Image.open(f'{path_to_dir}/{image_name}').convert("L")
            hsi.append(np.array(img))
        self.hsi = np.array(hsi).transpose((1, 2, 0))


    def save_to_mat(self, path_to_file: str, mat_key: str):
        """
        Saves HSI to .mat file as dictionary

        Parameters
        ----------
        path_to_file: str
            Path to saving file
        mat_key: str
            Key for dictionary
        """
        temp_dict = {mat_key: self.hsi}
        savemat(path_to_file, temp_dict)

    def save_to_tiff(self, path_to_file: str):
        """
        Saves HSI to .tiff file

        Parameters
        ----------
        path_to_file: str
            Path to saving file
        """
        ...

    def save_to_h5(self, path_to_file: str, h5_key: str):
        """
        Saves HSI to .h5 file as dictionary

        Parameters
        ----------
        path_to_file: str
            Path to saving file
        h5_key: str
            Key for dictionary
        """
        with h5py.File(path_to_file, 'w') as f:
            f.create_dataset(h5_key, data=self.hsi)

    def save_to_npy(self, path_to_file: str):
        """
        Saves HSI to .npy file

        Parameters
        ----------
        path_to_file: str
            Path to saving file
        """
        np.save(path_to_file, self.hsi)

    def save_to_images(self, path_to_dir: str, format: str = 'png'):
        """
        Saves HSI to .npy file

        Parameters
        ----------
        path_to_dir: str
            Path to saving file
        format: str
            Format of images (png, jpg, jpeg, bmp)
        """
        if not isdir(path_to_dir):
            mkdir(path_to_dir)
        for i in range(self.hsi.shape[-1]):
            if format in ('png', 'jpg', 'jpeg', 'bmp'):
                Image.fromarray(self.hsi[:, :, i]).convert("L").save(f'{path_to_dir}/{i}.{format}')
            else:
                raise Exception('Unexpected format')
