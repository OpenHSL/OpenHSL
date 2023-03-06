import numpy as np
from typing import Union
def normalization(mask: np.ndarray) -> np.ndarray:
    """
    normalization(mask)

        Returns a normalized mask from 0 to 1

        Parameters
        ----------
        mask: np.ndarray
            unnormalized array
        Return 
        ------
            np.ndarray
    """

    return (mask - mask.min()) / (mask.max() - mask.min())


def get_band_numbers(w_l: int, w_data: Union[list, np.ndarray]) -> int:
    """
    get_band_numbers(w_l, w_data)

        Returns a tuple of two channel values with minimum and maximum wavelength values

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


def ndvi_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    ndvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths
                
        Returns 
        ------
            np.ndarray        
    """
    
    wl_680 = 680
    wl_800 = 800

    red_band_numbers = get_band_numbers(wl_680, w_data)
    nir_band_numbers = get_band_numbers(wl_800, w_data)

    red = cube[:, :, red_band_numbers]
    nir = cube[:, :, nir_band_numbers]

    mask = (nir - red) / (nir + red) + 1
    mask[nir + red == 0] = 0

    return normalization(mask)


def dvi_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    dvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the DVI index 
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """
    
    wl_700 = 700
    wl_800 = 800

    band_numbers_700 = get_band_numbers(wl_700,  w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_700 = cube[:, :, band_numbers_700]
    channel_800 = cube[:, :, band_numbers_800]

    mask = channel_800 - channel_700

    return normalization(mask)


def osavi_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    osavi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'osavi' index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    wl_670 = 670
    wl_800 = 800

    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_670 = cube[:, :, band_numbers_670]
    channel_800 = cube[:, :, band_numbers_800]

    mask = 1.16 * (channel_800 - channel_670) / (channel_800 + channel_670 + 0.16)
    mask[channel_800 + channel_670 + 0.16 == 0] = 0

    return normalization(mask)


def sr_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    sr_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'SR' index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube[:, :, band_numbers_680]
    channel_800 = cube[:, :, band_numbers_800]

    mask = channel_800 / channel_680
    mask[channel_680 == 0] = 0

    return normalization(mask)


def wdrvi_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    wdrvi_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'wdrvi' index
        
        Parameters
        ----------
        cube: np.ndarrays
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube[:, :, band_numbers_680]
    channel_800 = cube[:, :, band_numbers_800]

    mask = (0.05 * channel_800 - channel_680)/(0.05 * channel_800 + channel_680)
    mask[0.05 * channel_800 + channel_680 == 0] = 0

    return normalization(mask)


def mtvi2_mask(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    mtvi2_mask(cube, w_data)
        #TODO добавить ссылку на описание индекса
        Calculating the 'mtvi2' index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    wl_550 = 550
    wl_670 = 670
    wl_800 = 800

    band_numbers_550 = get_band_numbers(wl_550, w_data)
    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_550 = cube[:, :, band_numbers_550]
    channel_670 = cube[:, :, band_numbers_670]
    channel_800 = cube[:, :, band_numbers_800]

    a = 1.5 * (1.2 * (channel_800 - channel_550) - 2.5 * (channel_670 - channel_550))
    b = np.sqrt((2 * channel_800 + 1)**2 - (6 * channel_800 - 5 * np.sqrt(channel_670)) - 0.5)
    mask = a * b
    mask[np.isnan(mask) == True] = 0

    return normalization(mask)

def simple_hsi_to_rgb(cube: np.ndarray, w_data: Union[list, np.ndarray]) -> np.ndarray:
    """
    hsi_for_rgb(cube, w_data)

        Return rgb-image from hyperspectral image
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns 
        ------
            np.ndarray        
    """

    wl_440 = 440
    wl_550 = 550
    wl_640 = 640

    blue_band_numbers = get_band_numbers(wl_440, w_data)
    green_band_numbers = get_band_numbers(wl_550, w_data)
    red_band_numbers = get_band_numbers(wl_640, w_data)

    blue = cube[:, :, blue_band_numbers]
    green = cube[:, :, green_band_numbers]
    red = cube[:, :, red_band_numbers]

    return np.dstack((blue, green, red))
