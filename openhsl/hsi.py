import os.path

import h5py
import rasterio
import numpy as np
from os import listdir, mkdir
from os.path import isdir, splitext
from PIL import Image
from scipy.io import loadmat, savemat
from typing import Optional, List
import json


class HSImage:
    """
    HSImage(hsi, metadata)

        Hyperspectral Image which has a dimension X - Y - Z
        where Z is a count of channels.
        Data are captured along axis X.

        Parameters
        ----------
        hsi: np.ndarray
            3D-matrix which has a dimension X - Y - Z.
            where:
                X is along-axis of capturing data.
                Y is constant resolution.
                Z is a count of channels.

        wavelengths: list
            contains set of wavelengths for each layer HS
            len(wavelengths) == hsi.shape[2] !!!

        Attributes
        ----------
        data: np.ndarray

        wavelengths: list

        Examples
        --------
            arr = np.zeros((100, 100, 250))
            wavelengths = [400, 402, 404, ..., 980]

            hsi = HSImage(hsi=arr, wavelengths=wavelengths)

    """

    def __init__(self,
                 hsi: Optional[np.ndarray] = None,
                 wavelengths: Optional[List] = None):
        """
            Inits HSI object.

        """
        if hsi is None:
            print('Created void HSI data')
        self.data = hsi
        if wavelengths is None:
            print('Wavelengths data is empty')
        self.wavelengths = wavelengths

    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, item):
        if item < len(self):
            return self.data[:, :, item]
        else:
            raise IndexError(f"{item} is too much for {len(self)} channels in hsi")
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.data.shape[-1]
    # ------------------------------------------------------------------------------------------------------------------

    def calibrate_white_reference(self, coefficients):
        self.data = self.data / coefficients[None, None, :]
    # ------------------------------------------------------------------------------------------------------------------

    def to_spectral_list(self):
        """
        Converts HSI to list of spectrals (as ravel)

        ^ y
        | [0][1][2]
        | [2][3][4] --> [1][2][3][4][5][6][7][8][9]
        | [5][6][7]
        --------> x

        Returns
        -------
        list
        """
        return np.reshape(self.data, (self.data.shape[0] * self.data.shape[1], self.data.shape[2]))
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_spectral_list(self, spectral_list, height, width):
        """
        Create HSI from spectral list with height and width


                                          ^ y
                                          | [0][1][2]
        [1][2][3][4][5][6][7][8][9] -->   | [2][3][4] -> [1][2][3][4][5][6][7][8][9]
                                          | [5][6][7]
                                          --------> x

        Parameters
        ----------
        spectral_list
        height
        width

        Returns
        -------

        """
        self.data = np.reshape(spectral_list, (height, width, len(spectral_list[0])))
    # ------------------------------------------------------------------------------------------------------------------

    def get_hyperpixel_by_coordinates(self,
                                      x: int,
                                      y: int) -> np.ndarray:
        """
        get_hyperpixel_by_coordinates(x, y)

            Returns hyperpixel from HSI by coordinates

            Parameters
            ----------
            x - X-coordinate
            y - Y-coordinate

            Returns
            -------
            np.ndarray
        """
        height, width, _ = self.data.shape
        if y >= height or x >= width:
            raise IndexError('Coordinates are out of range')
        return self.data[y, x, :]
    # ------------------------------------------------------------------------------------------------------------------

    def rot90(self):
        """
        rot90()

            Rotates for 90 degree hsi built-in counterclockwise
        """
        self.data = np.rot90(self.data, axes=(0, 1))
    # ------------------------------------------------------------------------------------------------------------------

    def load_metadata(self,
                      path_to_file: str):
        path_to_file = '.'.join(path_to_file.split('.')[:-1]) + '_metainfo.json'
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r') as json_file:
                data = json.load(json_file)
            self.wavelengths = data['wavelengths']
        else:
            print("Metainfo file does not exist! Wavelengths will be empty.")
            self.wavelengths = []
    # ------------------------------------------------------------------------------------------------------------------

    def save_metadata(self,
                      path_to_file: str):
        """
        save_metadata(path_to_file)

            Parameters
            ----------
            path_to_file: str
                path to json file

            Returns
            -------

        """
        path_to_file = '.'.join(path_to_file.split('.')[:-1]) + '_metainfo.json'
        if not self.wavelengths:
            print('Wavelengths are empty! Save as empy list.')
            self.wavelengths = []
        data = {"wavelengths": list(self.wavelengths)}
        with open(path_to_file, 'w') as outfile:
            outfile.write(json.dumps(data))
    # ------------------------------------------------------------------------------------------------------------------

    def load(self,
             path_to_data: str,
             key: Optional[str] = None):
        """
        load(path_to_data, key)

            Loading HSI from files

            Parameters
            ----------
            path_to_data: str
                path to data source such as directory (set of images) or file (mat, h5, npy, tiff)
            key: str
                key for files like mat or h5
        """
        if os.path.isdir(path_to_data):
            self.load_from_layer_images(path_to_dir=path_to_data)
        elif path_to_data.endswith('.mat'):
            self.load_from_mat(path_to_file=path_to_data, mat_key=key)
        elif path_to_data.endswith('.h5'):
            self.load_from_h5(path_to_file=path_to_data, h5_key=key)
        elif path_to_data.endswith('.npy'):
            self.load_from_npy(path_to_file=path_to_data)
        elif path_to_data.endswith('.tiff') or path_to_data.endswith('.tif'):
            self.load_from_tiff(path_to_file=path_to_data)
        else:
            raise Exception('Unsupported file extension')
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_mat(self,
                      path_to_file: str,
                      mat_key: str):
        """
        load_from_mat(path_to_file, mat_key)

            Loads HSI from .mat file.

            Parameters
            ----------
            path_to_file: str
                Path to .mat file
            mat_key: str
                Key for field in .mat file as dict object
                mat_file['image']
            Raises
            ------

        """
        self.data = loadmat(path_to_file)[mat_key]

        self.load_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_tiff(self,
                       path_to_file: str):
        """
        load_from_tiff(path_to_file)

            Loads HSI from .tiff file.

            Parameters
            ----------
            path_to_file: str
                Path to .tiff file
            """
            # TODO GDAL or what?
        raster = rasterio.open(path_to_file)
        band = raster.read()
        self.data = band.transpose((1, 2, 0))
        self.load_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_npy(self,
                      path_to_file: str):
        """
        load_from_npy(path_to_file)

            Loads HSI from .npy file.

            Parameters
            ----------
            path_to_file: str
                Path to .npy file
        """
        self.data = np.load(path_to_file)

        self.load_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_h5(self,
                     path_to_file: str,
                     h5_key: Optional[str] = None):
        """
        load_from_h5(path_to_file, h5_key)

            Loads HSI from .h5 file.

            Parameters
            ----------
            path_to_file: str
                Path to .h5 file
            h5_key: str
                Key for field in .h5 file as dict object
        """
        self.data = np.array(h5py.File(path_to_file, 'r')[h5_key])

        self.load_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_layer_images(self,
                               path_to_dir: str):
        """
        load_from_images(path_to_dir)

            Loads HSI from images are placed in directory.

            Parameters
            ----------
            path_to_dir: str
                Path to directory with images
        """
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)

        images_list = listdir(path_to_dir)
        hsi = []

        for image_name in images_list:
            img = Image.open(f'{path_to_dir}/{image_name}').convert("L")
            hsi.append(np.array(img))
        if not hsi:
            raise Exception("Can't read files!")

        self.data = np.array(hsi).transpose((1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_mat(self,
                    path_to_file: str,
                    mat_key: str):
        """
        save_to_mat(path_to_file, mat_key)

            Saves HSI to .mat file as dictionary

            Parameters
            ----------
            path_to_file: str
                Path to saving file
            mat_key: str
                Key for dictionary
        """
        temp_dict = {mat_key: self.data}
        savemat(path_to_file, temp_dict)

        self.save_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_tiff(self,
                     path_to_file: str):
        """
        save_to_tiff(path_to_file)

            Saves HSI to .tiff file

            Parameters
            ----------
            path_to_file: str
                Path to saving file
        """
        if not path_to_file.endswith('.tif') and not path_to_file.endswith('.tiff'):
            raise Exception('Incorrect file format')

        dt = 'int8'
        if self.data.dtype.name == 'uint8' or self.data.dtype.name == 'int8':
            dt = 'int8'
        elif self.data.dtype.name == 'uint16' or self.data.dtype.name == 'int16':
            dt = 'int16'
        elif self.data.dtype.name == 'uint32' or self.data.dtype.name == 'int32':
            dt = 'int32'

        d = {'driver': 'GTiff',
             'dtype': dt,
             'nodata': None,
             'width': self.data.shape[1],
             'height': self.data.shape[0],
             'count': self.data.shape[2],
             # 'crs': CRS.from_epsg(32736),
             # 'transform': Affine(10.0, 0.0, 653847.1979372115, 0.0, -10.0, 7807064.5603836905),
             # 'tiled': False,
             'interleave': 'band'}

        with rasterio.open(path_to_file, 'w', **d) as dst:
            dst.write(self.data.transpose((2, 0, 1)))
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_h5(self,
                   path_to_file: str,
                   h5_key: str):
        """
        save_to_h5(path_to_file, h5_key)

            Saves HSI to .h5 file as dictionary.

            Parameters
            ----------
            path_to_file: str
                Path to saving file
            h5_key: str
                Key for dictionary
        """
        with h5py.File(path_to_file, 'w') as f:
            f.create_dataset(h5_key, data=self.data)

        self.save_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_npy(self,
                    path_to_file: str):
        """
        save_to_npy(path_to_file)

            Saves HSI to .npy file.

            Parameters
            ----------
            path_to_file: str
                Path to saving file
        """
        np.save(path_to_file, self.data)
        self.save_metadata(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_images(self,
                       path_to_dir: str,
                       format: str = 'png'):
        """
        save_to_images(path_to_dir, format)

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
        for i in range(self.data.shape[-1]):
            if format in ('png', 'jpg', 'jpeg', 'bmp'):
                Image.fromarray(self.data[:, :, i]).convert("L").save(f'{path_to_dir}/{i}.{format}')
            else:
                raise Exception('Unexpected format')
    # ------------------------------------------------------------------------------------------------------------------
