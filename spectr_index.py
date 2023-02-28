import numpy as np
def get_band_numbers(left_border: int, right_border: int, w_data: list) -> tuple:
    """
    get_band_numbers(left_border, right_border, w_data)

        Returns a tuple of two channel values with minimum and maximum wavelength values

        Parameters
        ----------
        left_border: int
           left border in nm

        right_border: int
            the right boundary in nm

        w_data: list
            list of wavelengths

        Return 
        ------
            list     
    """
    part_bands_list = [i for i, v in enumerate(w_data) if left_border < v < right_border]

    return min(part_bands_list), max(part_bands_list)


def reduce_mean(hsi: np.ndarray, l_bound: int, r_bound: int) -> np.ndarray:
    """
    reduceMean(HSI, l_bound, r_bound):

        Calculating average values from multiple channels
        
        Parameters
        ----------
        hsi: np.ndarray
           hyperspectral image

        l_bound: int
            channel number in the hyperspectral image, left border
            
        r_bound: int
            channel number in the hyperspectral image, right border

        Return 
        ------
            np.ndarray 

    """
    return np.mean(hsi[:, :, l_bound:r_bound], axis=2)


def ndvi_mask(cube: np.ndarray,
              w_data: list,
              left_red=633,
              right_red=650,
              left_nir=844,
              right_nir=860) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        left_red: int 
            the left border of red in nanometers

        right_red: int
            the right border of red in nanometers

        left_nir: int 
            the left border of nir in nanometers
            
        right_nir: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """

    red_band_numbers = get_band_numbers(left_red, right_red, w_data)
    nir_band_numbers = get_band_numbers(left_nir, right_nir, w_data)

    red = reduce_mean(cube, red_band_numbers[0], red_band_numbers[1])
    nir = reduce_mean(cube, nir_band_numbers[0], nir_band_numbers[1])

    mask = (nir - red) / (nir + red) + 1
    mask[nir + red == 0] = 0
    
    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())


def dvi_mask(cube: np.ndarray,
            w_data: list,
            left_700=695,
            right_700=705,
            left_800=795,
            right_800=805) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        left_700: int 
            the left border of red in nanometers

        right_700: int
            the right border of red in nanometers

        left_800: int 
            the left border of nir in nanometers
            
        right_800: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """
    
    band_numbers_700 = get_band_numbers(left_700, right_700, w_data)
    band_numbers_800 = get_band_numbers(left_800, right_800, w_data)

    channel_700 = reduce_mean(cube, band_numbers_700[0], band_numbers_700[1])
    channel_800 = reduce_mean(cube, band_numbers_800[0], band_numbers_800[1])

    mask = (channel_800 - channel_700)

    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())


def osavi_mask(cube: np.ndarray,
            w_data: list,
            left_670=665,
            right_670=675,
            left_800=795,
            right_800=805) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        left_670: int 
            the left border of red in nanometers

        right_670: int
            the right border of red in nanometers

        left_800: int 
            the left border of nir in nanometers
            
        right_800: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_670 = get_band_numbers(left_670, right_670, w_data)
    band_numbers_800 = get_band_numbers(left_800, right_800, w_data)

    channel_670 = reduce_mean(cube, band_numbers_670[0], band_numbers_670[1])
    channel_800 = reduce_mean(cube, band_numbers_800[0], band_numbers_800[1])
    
    print("-----wdrvi_mask-----") 
    print('channel_670 :',np.unique(channel_670))
    print("")
    print('channel_800 :',np.unique(channel_800))
    print("")

    mask = 1.16*(channel_800 - channel_670)/(channel_800 + channel_670 +0.16)
    mask[channel_800 + channel_670 +0.16 == 0] = 0

    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())


def sr_mask(cube: np.ndarray,
            w_data: list,
            left_680=675,
            right_680=685,
            left_800=795,
            right_800=805) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the SR index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        left_680: int 
            the left border of red in nanometers

        right_680: int
            the right border of red in nanometers

        left_800: int 
            the left border of nir in nanometers
            
        right_800: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """
    
    band_numbers_680 = get_band_numbers(left_680, right_680, w_data)
    band_numbers_800 = get_band_numbers(left_800, right_800, w_data)

    print("-----band_numbers-----") 
    print('band_numbers_700 :',np.unique(band_numbers_680))
    print("")
    print('band_numbers_800 :',np.unique(band_numbers_800))
    print("")

    channel_680 = reduce_mean(cube, band_numbers_680[0], band_numbers_680[1])
    channel_800 = reduce_mean(cube, band_numbers_800[0], band_numbers_800[1])
    
    mask = channel_800/channel_680
    mask[channel_680 == 0] = 0

    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())


def wdrvi_mask(cube: np.ndarray,
            w_data: list,
            left_680=675,
            right_680=685,
            left_800=795,
            right_800=805) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the wdrvi index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        left_680: int 
            the left border of red in nanometers

        right_680: int
            the right border of red in nanometers

        left_800: int 
            the left border of nir in nanometers
            
        right_800: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_680 = get_band_numbers(left_680, right_680, w_data)
    band_numbers_800 = get_band_numbers(left_800, right_800, w_data)

    channel_680 = reduce_mean(cube, band_numbers_680[0], band_numbers_680[1])
    channel_800 = reduce_mean(cube, band_numbers_800[0], band_numbers_800[1])

    mask = (0.05*channel_800 - channel_680)/(0.05*channel_800 + channel_680)
    mask[0.05*channel_800 + channel_680 == 0] = 0

    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())


def mtvi2_mask(cube: np.ndarray,
            w_data: list,
            left_550=545,
            right_550=555,
            left_670=665,
            right_670=675,
            left_800=795,
            right_800=805) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the mtvi2 index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths
        
        left_550: int 
            the left border of  green in nanometers

        right_550: int
            the right border of green in nanometers

        left_670: int 
            the left border of red in nanometers

        right_670: int
            the right border of red in nanometers

        left_800: int 
            the left border of nir in nanometers
            
        right_800: int
            the right border of nir in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_550 = get_band_numbers(left_550, right_550, w_data)
    band_numbers_670 = get_band_numbers(left_670, right_670, w_data)
    band_numbers_800 = get_band_numbers(left_800, right_800, w_data)

    channel_550 = reduce_mean(cube, band_numbers_550[0], band_numbers_550[1])
    channel_670 = reduce_mean(cube, band_numbers_670[0], band_numbers_670[1])
    channel_800 = reduce_mean(cube, band_numbers_800[0], band_numbers_800[1])

    a = 1.5*(1.2 * (channel_800 - channel_550) - 2.5 * (channel_670 - channel_550))
    b = np.sqrt((2 * channel_800 + 1)**2 - (6 * channel_800 - 5*np.sqrt((channel_670))) - 0.5)
    mask = a*b

    #Значения сделаю пока от 0 до 1
    return (mask - mask.min())/(mask.max() - mask.min())