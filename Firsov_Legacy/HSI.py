import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.io import loadmat, savemat


class HSImage:
    """
    Hyperspectral Image has dimension X - Y - Z where Z - count of channels
    Attributes
    ----------
    hsi : np.array
        Hyperspectral Image in array format
    coef : np.array
        Coefficients matrix for normalizing input spectrum if slit has defects
    labels : dict
        Dictionary of classes in mask
        i.e. {0: "void", 1: "background", 2: "class_1", 3: "class_2"}
    Methods
    ---------
    TODO write all methods
    """

    def __init__(self, hsi=None, coef=None, labels=None):
        self.hsi = hsi
        self.coef = coef
        self.labels = labels

    def _coef_norm(self, hs_layer: np.array, thresh=100) -> np.array:
        """
        This method calculates matrix of normalize coefficients from spectrum layer obtained from slit
        Parameters
        ----------
        hs_layer : np.array
            Layer from hyperspectral image obtained from raw slit
        thresh : int
            Value to which whole spectrum will be normalized
        """
        coef = []
        for i in range(250):
            coef.append([x / thresh for x in hs_layer[:, i]])
        return np.array(coef).T

    def set_coef(self, path_to_norm: str=None, key: str=None):
        """
        This method set coefficients for normalize HSI from file with .mat or .tiff extension
        Parameters
        ----------
        path_to_norm : str
            path to file with raw spectrum obtained from slit
        key : str
            key from .mat file
        """
        if path_to_norm:
            if path_to_norm.endswith('.mat'):
                if key:
                    temp = loadmat(path_to_norm)[key]
                    self.coef = self._coef_norm(temp[:, 5, :])
            if path_to_norm.endswith('.tiff'):
                temp = tiff.imread(path_to_norm)
                self.coef = self._coef_norm(temp[:, 5, :])

    def set_labels(self, labels):
        self.labels = labels
    def _crop_layer(self, layer: np.array,
                    gap_coord=620,
                    range_to_spectrum=185,
                    range_to_end_spectrum=250,
                    left_bound_spectrum=490,
                    right_bound_spectrum=1390
                    ) -> np.array:
        """
        This method crops layer to target area which contain spectrum and return it
        Parameters
        ----------
        layer : np.array
            layer of HSI
        gap_coord : int
            it means coordinate of line from diffraction slit
        range_to_spectrum : int
            range from diffraction slit line to area of spectrum
        range_to_end_spectrum : int
            width of spectrum line
        left_bound_spectrum and right_bound spectrum : int
            boundaries of where is spectrum
        """
        x1 = gap_coord + range_to_spectrum
        x2 = x1 + range_to_end_spectrum
        return layer[x1: x2, left_bound_spectrum: right_bound_spectrum].T


    def _normalize_spectrum_layer(self, layer: np.array,
                                  coef=None,
                                  ) -> np.array:
        """
        This method normalizes layer with not uniform light
        Parameters
        ----------
        layer : np.array
            layer of HSI
        coef : np.array
            array of coefficients for uniform light
        """
        return layer / coef if coef else layer

    def _prepare_layer(self, layer: np.array) -> np.array:
        """
        This method crops and normalizes input layer of spectrum and return it
        Parameters
        ----------
        layer : np.array
            layer of HSI
        """
        layer = self._crop_layer(layer)
        layer = self._normalize_spectrum_layer(layer, self.coef)
        return layer

    def add_layer_yz_fast(self, layer: np.array, i: int, count_images: int):
        """
        This method adds layer for X-coordinate with preallocated memory to hyperspectral image
        Parameters
        ----------
        layer : np.array
            layer of HSI
        i : int
            index of current layer
        count_images : int
            length HSI by X-coordinate (count of layers)
        """
        layer = self._prepare_layer(layer)
        if (self.hsi is None):
            x, y, z = count_images, *(layer.shape)
            self.hsi = np.zeros((x, y, z))
        # TODO squeeze
        self.hsi[i, :, :] = layer[None, :, :]

    def add_layer_yz(self, layer: np.array):
        """
        This method adds layer for X-coordinate
        Parameters
        ----------
        layer : np.array
            layer of HSI
        """
        layer = self._prepare_layer(layer)
        if (self.hsi is None):
            self.hsi = layer
        elif (len(np.shape(self.hsi)) < 3):
            self.hsi = np.stack((self.hsi, layer), axis=0)
        else:
            self.hsi = np.append(self.hsi, layer[None, :, :], axis=0)

    def add_layer_xy(self, layer: np.array):
        """
        This method adds layer as image to HSI for Z-coordinate
        Parameters
        ----------
        layer : np.array
            layer of HSI as image
        """
        if (self.hsi is None):
            self.hsi = layer
        elif (len(np.shape(self.hsi)) < 3):
            self.hsi = np.stack((self.hsi, layer), axis=2)
        else:
            self.hsi = np.append(self.hsi, layer[:, :, None], axis=2)

    # TODO make
    def rgb(self, channels=(80, 70, 20)) -> np.array:
        """
        This method transforms hyperspectral image to RGB image
        Parameters
        ----------
        channels : tuple[red: int, green: int, blue: int]
            Tuple of numbers of channels accorded to wavelengths of red, green and blue colors
        """
        r, g, b = channels
        return np.stack((self.hsi[:, :, r], self.hsi[:, :, g], self.hsi[:, :, b]), axis=2)

    def hyp_to_mult(self, number_of_channels: int) -> np.array:
        """
        Converts hyperspectral image to multispectral and return it
        Parameters
        ----------
        number_of_channels : int
            number of channels of multi-spectral image
        """
        if (number_of_channels > np.shape(self.hsi)[2]):
            raise ValueError('Number of MSI is over then HSI')
        MSI = np.zeros((np.shape(self.hsi)[0], np.shape(self.hsi)[1], number_of_channels))
        l = [int(x * (250 / number_of_channels)) for x in range(0, number_of_channels)]
        for k, i in enumerate(l):
            MSI[:, :, k] = self.hsi[:, :, i]

        return MSI

    def save_channel_as_png(self, path_to_png: str, num_channel: int, color_mode='gray'):
        plt.imsave(path_to_png, self.get_channel(num_channel), cmap=color_mode)

    def get_hsi(self) -> np.array:
        """
        Returns current hyperspectral image as array
        """
        return self.hsi

    def get_labels(self) -> dict:
        """
        Returns dictionary of labels of HSImage
        """
        return self.labels

    def get_channel(self, number_of_channel: int) -> np.array:
        """
        Returns channel of hyperspectral image
        Parameters
        ----------
        number_of_channel : int
            number of channel of hyperspectral image
        """
        return self.hsi[:, :, number_of_channel]

    def load_from_array(self, hsi: np.array):
        """
        Initializes hyperspectral image from numpy array
        Parameters
        ----------
        hsi : np.array
            hyperspectral image as array
        """
        self.hsi = hsi

    def load_from_mat(self, path_to_file: str, key: str):
        """
        Initializes hyperspectral image from mat file
        Parameters
        ----------
        path_to_file : str
            path to .mat file with HSI
        key : str
            key for dictionary in .mat file
        """
        self.hsi = loadmat(path_to_file)[key]
        try:
            self.labels = loadmat(path_to_file)['labels']
        except:
            print('This file has not labels')

    def save_to_mat(self, path_to_file: str, key: str):
        """
        Saves hyperspectral image to .mat file
        Parameters
        ----------
        path_to_file : str
            path to .mat file with HSI
        key : str
            key for dictionary in .mat file
        """
        if(not path_to_file.endswith('.mat')):
            path_to_file += '.mat'

        # TODO Check values in raw images
        if self.labels:
            savemat(path_to_file, {key: self.hsi.astype('int16'), 'labels': self.labels})
        else:
            savemat(path_to_file, {key: self.hsi.astype('int16')})
    def load_from_tiff(self, path_to_file: str):
        """
        Initializes hyperspectral image from .tiff file
        Parameters
        ----------
        path_to_file : str
            path to .tiff file with HSI
        """
        self.hsi = tiff.imread(path_to_file)

    def save_to_tiff(self, path_to_file):
        pass

    def load_from_npy(self, path_to_file: str):
        """
        Initializes hyperspectral image from .npy file
        Parameters
        ----------
        path_to_file : str
            path to .npy file with HSI
        """
        self.hsi = np.load(path_to_file)

    def save_to_npy(self, path_to_file):
        """
        Saves hyperspectral image to .npy file
        Parameters
        ----------
        path_to_file : str
            path to .npy file with HSI
        """
        np.save(path_to_file, self.hsi)
