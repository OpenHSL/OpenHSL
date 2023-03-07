import numpy as np
from typing import Union
from hsi import HSImage
from scipy.interpolate import interp1d


def normalization(mask: np.ndarray) -> np.ndarray:
    """
    normalization(mask)

        Returns a normalized mask from 0 to 1

        Parameters
        ----------
        mask: np.ndarray
            Unnormalized array
        Return 
        ------
            np.ndarray
    """

    return (mask - mask.min()) / (mask.max() - mask.min())


def neighbor_el(elements_list: list, element: float) -> float:
    """
    neighbor_el(l, el)
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


def get_band_numbers(w_l: int, w_data: Union[list, np.ndarray]) -> int:
    """
    get_band_numbers(w_l, w_data)

        Returns a tuple of two channel values with minimum and maximum wavelength values
        # TODO update this docstring: not actual variables names
        Parameters
        ----------
        left_border: int
           left border in nm

        right_border: int
            the right boundary in nm

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


def ndvi_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    ndvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths
                
        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_680 = 680
    wl_800 = 800

    red_band_numbers = get_band_numbers(wl_680, w_data)
    nir_band_numbers = get_band_numbers(wl_800, w_data)

    red = cube_data[:, :, red_band_numbers]
    nir = cube_data[:, :, nir_band_numbers]

    mask = (nir - red) / (nir + red) + 1
    mask[nir + red == 0] = 0

    return normalization(mask)


def dvi_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    dvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the DVI index 
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_700 = 700
    wl_800 = 800

    band_numbers_700 = get_band_numbers(wl_700, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_700 = cube_data[:, :, band_numbers_700]
    channel_800 = cube_data[:, :, band_numbers_800]

    mask = channel_800 - channel_700

    return normalization(mask)


def osavi_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    osavi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'osavi' index
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_670 = 670
    wl_800 = 800

    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_670 = cube_data[:, :, band_numbers_670]
    channel_800 = cube_data[:, :, band_numbers_800]

    mask = 1.16 * (channel_800 - channel_670) / (channel_800 + channel_670 + 0.16)
    mask[channel_800 + channel_670 + 0.16 == 0] = 0

    return normalization(mask)


def sr_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    sr_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'SR' index
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube_data[:, :, band_numbers_680]
    channel_800 = cube_data[:, :, band_numbers_800]

    mask = channel_800 / channel_680
    mask[channel_680 == 0] = 0

    return normalization(mask)


def wdrvi_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    wdrvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'wdrvi' index
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube_data[:, :, band_numbers_680]
    channel_800 = cube_data[:, :, band_numbers_800]

    mask = (0.05 * channel_800 - channel_680) / (0.05 * channel_800 + channel_680)
    mask[0.05 * channel_800 + channel_680 == 0] = 0

    return normalization(mask)


def mtvi2_mask(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    mtvi2_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'mtvi2' index
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_550 = 550
    wl_670 = 670
    wl_800 = 800

    band_numbers_550 = get_band_numbers(wl_550, w_data)
    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_550 = cube_data[:, :, band_numbers_550]
    channel_670 = cube_data[:, :, band_numbers_670]
    channel_800 = cube_data[:, :, band_numbers_800]

    a = 1.5 * (1.2 * (channel_800 - channel_550) - 2.5 * (channel_670 - channel_550))
    b = np.sqrt((2 * channel_800 + 1) ** 2 - (6 * channel_800 - 5 * np.sqrt(channel_670)) - 0.5)
    mask = a * b
    mask[np.isnan(mask) == True] = 0

    return normalization(mask)


def simple_hsi_to_rgb(cube: Union[HSImage, np.ndarray], w_data: Union[list, np.ndarray] = False) -> np.ndarray:
    """
    hsi_for_rgb(cube, w_data)

        Return rgb-image from hyperspectral image
        
        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    if type(cube) == HSImage:
        cube_data = cube.data
        w_data = cube.wavelengths

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wl_440 = 440
    wl_550 = 550
    wl_640 = 640

    blue_band_numbers = get_band_numbers(wl_440, w_data)
    green_band_numbers = get_band_numbers(wl_550, w_data)
    red_band_numbers = get_band_numbers(wl_640, w_data)

    blue = cube_data[:, :, blue_band_numbers]
    green = cube_data[:, :, green_band_numbers]
    red = cube_data[:, :, red_band_numbers]

    return np.dstack((blue, green, red))


def XYZ2sRGB_exgamma(XYZ):
    """
    See IEC_61966-2-1.pdf
    No gamma correction has been incorporated here, nor any clipping, so this
    transformation remains strictly linear.  Nor is there any data-checking.
    DHF 9-Feb-11
    """
    # Image dimensions
    d = XYZ.shape
    r = d[0] * d[1]
    w = d[2]

    # Reshape for calculation, converting to w columns with r rows.
    XYZ = np.reshape(XYZ, (r, w))

    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).

    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0414],
                  [0.0557, -0.2040, 1.0570]])

    sRGB = np.dot(XYZ, M.T)

    # Reshape to recover shape of original input.
    sRGB = np.reshape(sRGB, d)

    return sRGB


# TODO finalize it.
def hsi_to_rgb(cube: Union[HSImage, np.ndarray], illum_coef: np.ndarray, xyzbar: np.ndarray) -> np.ndarray:
    """
    hsi_to_rgb(cube, illum_coef, xyzbar)

        Extracts an RGB image from an HSI image

        Parameters
        ----------
        cube: HSImage or np.ndarray
            hyperspectral image

        illum_coef: np.ndarray
            spectral intensity coefficients of the light source
        xyzbar: np.ndarray
            color space coefficients CIE 1931

        Returns 
        ------
            np.ndarray 


    """

    if type(cube) == HSImage:
        cube_data = cube.data

    elif type(cube) == np.ndarray:
        cube_data = cube
        del cube

    wavelengths = list(np.linspace(400, 720, 33))

    wavelengths_cube = list(np.linspace(420, 980, cube_data.shape[-1]))
    right_bound = wavelengths_cube.index(neighbor_el(wavelengths_cube, 720))
    rgb_waves = wavelengths_cube[:right_bound]

    # Функция которая в видимом диапазоне
    f_illum = interp1d(wavelengths, illum_coef[:, 0])

    new_cube = np.zeros((np.shape(cube_data)[0], np.shape(cube_data)[1], right_bound))
    for i, w in zip(range(new_cube.shape[-1]), rgb_waves):
        new_cube[:, :, i] = cube_data[:, :, i] * f_illum(w)

    r, c, w = new_cube.shape
    radiances = np.reshape(new_cube, (r * c, w))
    radiances.shape

    xyzbar_0 = xyzbar[:, 0]
    xyzbar_1 = xyzbar[:, 1]
    xyzbar_2 = xyzbar[:, 2]

    f_0 = interp1d(wavelengths, xyzbar_0)
    f_1 = interp1d(wavelengths, xyzbar_1)
    f_2 = interp1d(wavelengths, xyzbar_2)

    xyz_0 = [f_0(i) for i in rgb_waves]
    xyz_1 = [f_1(i) for i in rgb_waves]
    xyz_2 = [f_2(i) for i in rgb_waves]

    xyzbar_new = (np.array([xyz_0, xyz_1, xyz_2])).T

    XYZ = np.dot(radiances, xyzbar_new)
    XYZ = np.reshape(XYZ, (r, c, 3))
    XYZ = (XYZ - np.min(XYZ)) / (np.max(XYZ) - np.min(XYZ))

    RGB = XYZ2sRGB_exgamma(XYZ)
    RGB = (RGB - np.min(RGB)) / (np.max(RGB) - np.min(RGB))

    gray_mean = np.mean(RGB, axis=2)
    un = np.unique(gray_mean)

    # Так оставить?
    thresh = 0.98
    coord = [np.where(gray_mean == un[int(len(un) * thresh - 1)])][0]
    x, y = coord
    m = max(RGB[int(x), int(y), :])

    RGB[RGB > m] = m
    RGB = RGB / np.max(RGB)

    return RGB
