import h5py
import json
import numpy as np
import os.path
import rasterio

from os import listdir, mkdir
from PIL import Image
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
from typing import List, Optional, Union


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
        """
        Returns i-channel of HSI

        Parameters
        ----------
        item

        Returns
        -------

        """
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

    def flip_wavelengths(self):
        self.data = np.flip(self.data, axis=2)
    # ------------------------------------------------------------------------------------------------------------------

    def to_spectral_list(self):
        """
        Converts HSI to list of spectrals (as ravel)

        ^ y
        | [0][1][2]
        | [3][4][5] --> [0][1][2][3][4][5][6][7][8]
        | [6][7][8]
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
        [0][1][2][3][4][5][6][7][8] -->   | [3][4][5]
                                          | [6][7][8]
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
        if self.wavelengths is None:
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

    def save(self,
             path_to_data: str,
             key=None,
             img_format=None):

        pth = os.path.dirname(path_to_data)
        if not os.path.exists(pth):
            os.mkdir(pth)

        if os.path.isdir(path_to_data):
            self.save_to_images(path_to_dir=path_to_data, img_format=img_format)
        elif path_to_data.endswith('.mat'):
            self.save_to_mat(path_to_file=path_to_data, mat_key=key)
        elif path_to_data.endswith('.h5'):
            self.save_to_h5(path_to_file=path_to_data, h5_key=key)
        elif path_to_data.endswith('.npy'):
            self.save_to_npy(path_to_file=path_to_data)
        elif path_to_data.endswith('.tiff') or path_to_data.endswith('.tif'):
            self.save_to_tiff(path_to_file=path_to_data)
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
        with rasterio.open(path_to_file) as raster:
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

        dt = 'uint8'
        if self.data.dtype.name == 'uint8' or self.data.dtype.name == 'int8':
            dt = 'uint8'
        elif self.data.dtype.name == 'uint16' or self.data.dtype.name == 'int16':
            dt = 'uint16'
        elif self.data.dtype.name == 'uint32' or self.data.dtype.name == 'int32':
            dt = 'uint32'

        d = {'driver': 'GTiff',
             'dtype': dt,
             'nodata': None,
             'width': self.data.shape[1],
             'height': self.data.shape[0],
             'count': self.data.shape[2],
             'interleave': 'band'}

        with rasterio.open(path_to_file, 'w', **d) as dst:
            dst.write(self.data.transpose((2, 0, 1)))
        self.save_metadata(path_to_file)
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
                       img_format: str = 'png'):
        """
        save_to_images(path_to_dir, format)

            Saves HSI to .npy file

            Parameters
            ----------
            path_to_dir: str
                Path to saving file
            img_format: str
                Format of images (png, jpg, jpeg, bmp)
        """
        if not os.path.isdir(path_to_dir):
            mkdir(path_to_dir)

        supported_formats = tuple(['png', 'jpg', 'jpeg', 'bmp'])

        if img_format in supported_formats:
            for i in range(self.data.shape[-1]):
                Image.fromarray(self.data[:, :, i]).convert("L").save(f'{path_to_dir}/{i}.{img_format}')
        else:
            raise Exception('Unexpected format')
    # ------------------------------------------------------------------------------------------------------------------


def __neighbor_el(elements_list: list, element: float) -> float:
    """
    neighbor_el(elements_list, element)

        Return the closest element from list to given element

        Parameters
        ----------
        elements_list: list

        element: float

        Returns
        -------
            float
    """
    return min(elements_list, key=lambda x: abs(x - element))
# ----------------------------------------------------------------------------------------------------------------------


def __get_band_numbers(w_l: int, w_data: Union[list, np.ndarray]) -> int:
    """
    get_band_numbers(w_l, w_data)

        Returns the required channel value in the hyperspectral image

        Parameters
        ----------
        w_l: int
           the desired wavelength (nm)

        w_data: list or np.ndarray
            list of wavelengths

        Returns
        ------
            int
    """

    if w_l in w_data:
        w_data = list(w_data)
        return w_data.index(w_l)
    else:
        w_data = np.array(w_data)
        delta = w_data - w_l
        abs_delta = list(map(abs, delta))
        index_new_wl = abs_delta.index(min(abs_delta))

        return index_new_wl
# ----------------------------------------------------------------------------------------------------------------------


def minmax_normalization(mask: np.ndarray) -> np.ndarray:
    """
    normalization(mask)

        Returns a normalized mask from 0 to 1

        Parameters
        ----------
        mask: np.ndarray
            Denormalized array
        Return
        ------
            np.ndarray
    """

    return (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
# ----------------------------------------------------------------------------------------------------------------------


def contrast_correction(rgb, gamma_thresh):
    gray_mean = np.mean(rgb, axis=2)
    un = np.unique(gray_mean)

    coord = np.where(gray_mean == un[int(len(un) * gamma_thresh - 1)])
    x, y = coord
    m = np.max(rgb[int(x[0]), int(y[0]), :])

    rgb[rgb > m] = m
    rgb = rgb / np.max(rgb)

    return rgb
# ----------------------------------------------------------------------------------------------------------------------


def __xyz2srgb_exgamma(xyz: np.ndarray) -> np.ndarray:
    """
    See IEC_61966-2-1.pdf
    No gamma correction has been incorporated here, nor any clipping, so this
    transformation remains strictly linear.  Nor is there any data-checking.
    DHF 9-Feb-11
    """
    # Image dimensions
    d = xyz.shape
    r = d[0] * d[1]
    w = d[2]

    # Reshape for calculation, converting to w columns with r rows.
    xyz = np.reshape(xyz, (r, w))

    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).

    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0414],
                  [0.0557, -0.2040, 1.0570]])

    s_rgb = np.dot(xyz, m.T)

    # Reshape to recover shape of original input.
    s_rgb = np.reshape(s_rgb, d)

    return s_rgb
# ----------------------------------------------------------------------------------------------------------------------


def __get_bounds_vlr(w_data: List):
    """
    Returns visible left and right spectrum bounds
    Parameters
    ----------
    w_data:
        list of wavelengths
    """
    right_bound = w_data.index(__neighbor_el(w_data, 720))
    left_bound = w_data.index(__neighbor_el(w_data, 400))
    return left_bound, right_bound
# ----------------------------------------------------------------------------------------------------------------------


def __convert_hsi_to_xyz(xyz_bar_path,
                         hsi,
                         rgb_waves):
    """
    Converting HSI to XYZ
    Parameters
    ----------
    xyz_bar_path
    hsi
    rgb_waves

    Returns
    -------

    """
    xyz_bar = loadmat(xyz_bar_path)['xyzbar']

    xyz_bar_0 = xyz_bar[:, 0]
    xyz_bar_1 = xyz_bar[:, 1]
    xyz_bar_2 = xyz_bar[:, 2]

    wl_vlr = np.linspace(400, 720, 33)

    f_0 = interp1d(wl_vlr, xyz_bar_0)
    f_1 = interp1d(wl_vlr, xyz_bar_1)
    f_2 = interp1d(wl_vlr, xyz_bar_2)

    xyz_0 = [f_0(i) for i in rgb_waves]
    xyz_1 = [f_1(i) for i in rgb_waves]
    xyz_2 = [f_2(i) for i in rgb_waves]

    xyz_bar_new = (np.array([xyz_0, xyz_1, xyz_2])).T

    r, c, w = hsi.shape
    radiances = np.reshape(hsi, (r * c, w))

    xyz = np.dot(radiances, xyz_bar_new)
    xyz = np.reshape(xyz, (r, c, 3))
    xyz = (xyz - np.min(xyz)) / (np.max(xyz) - np.min(xyz))
    return xyz
# ----------------------------------------------------------------------------------------------------------------------


def simple_hsi_to_rgb(hsi: HSImage,
                      gamma_thresh: float = 0.98) -> np.ndarray:
    """
    simple_hsi_to_rgb(cube, wave_data)

        Return rgb-image from hyperspectral image

        Parameters
        ----------
        hsi: HSImage or np.ndarray
           hyperspectral image

        gamma_thresh

        Returns
        ------
            np.ndarray
    """

    cube_data = hsi.data

    if hsi.wavelengths is None:
        raise Exception("Cannot convert HSI to RGB without wavelengths information")
    else:
        w_data = hsi.wavelengths

    wl_440 = 440
    wl_550 = 550
    wl_640 = 640

    blue_band_numbers = __get_band_numbers(wl_440, w_data)
    green_band_numbers = __get_band_numbers(wl_550, w_data)
    red_band_numbers = __get_band_numbers(wl_640, w_data)

    blue = cube_data[:, :, blue_band_numbers].astype(float)
    green = cube_data[:, :, green_band_numbers].astype(float)
    red = cube_data[:, :, red_band_numbers].astype(float)

    simple_rgb = np.dstack((red.astype(np.uint8), green.astype(np.uint8), blue.astype(np.uint8)))

    simple_rgb = contrast_correction(simple_rgb, gamma_thresh)

    return simple_rgb
# ----------------------------------------------------------------------------------------------------------------------


def hsi_to_rgb(hsi: HSImage,
               xyz_bar_path: str = './xyzbar.mat',
               gamma_thresh: float = 0.98) -> np.ndarray:
    """
    hsi_to_rgb(cube, w_data, illumination_coef, xyzbar)

        Extracts an RGB image from an HSI image

        Parameters
        ----------
        hsi: HSImage or np.ndarray
            hyperspectral image

        xyz_bar_path: str
            path to mat file with CMF CIE 1931

        gamma_thresh: float
            coefficient for contrast correction

        Returns
        ------
            np.ndarray

    """

    hsi_data = hsi.data

    if hsi.wavelengths is None:
        raise Exception("Cannot convert HSI to RGB without wavelengths information")
    else:
        w_data = list(hsi.wavelengths)

    left_bound, right_bound = __get_bounds_vlr(w_data)

    rgb_waves = w_data[left_bound: right_bound]

    new_cube = hsi_data[:, :, left_bound: right_bound]

    xyz = __convert_hsi_to_xyz(xyz_bar_path=xyz_bar_path,
                               hsi=new_cube,
                               rgb_waves=rgb_waves)

    rgb = __xyz2srgb_exgamma(xyz)
    rgb = minmax_normalization(rgb)

    rgb = contrast_correction(rgb, gamma_thresh)

    return rgb
# ----------------------------------------------------------------------------------------------------------------------
