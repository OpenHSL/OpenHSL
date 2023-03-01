import numpy as np

def normalization(mask: np.ndarray) -> np.ndarray:
    """
    normalization(mask: np.ndarrya)

        Returns a normalized mask from 0 to 1

        Parameters
        ----------
        mask: np.ndarrya
            normalized mask
        Return 
        ------
            np.ndarray
    """
    return (mask - mask.min())/(mask.max() - mask.min())


def get_band_numbers(w_l: int, w_data: list) -> int:
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

    #delta:  [...-4 -2  0  3...]) Отклонение от нужной длины волны
    delta = w_data - w_l
    
    #определяем минимальный положительный элемент из списка отклонений
    min_plus_delta = min(delta[delta >= 0])

    #определяем индекс этого элемента в списке дельт
    index_min_poloj_delta = np.where(delta==min_plus_delta)
    
    #определили длину волны для подсчета индекса
    wat_len = w_data[index_min_poloj_delta]
    
    #канал по длине волны
    band_number = np.where(w_data == wat_len)
   
    return band_number[0]


def ndvi_mask(cube: np.ndarray,
            w_data: list,
            wl_680 = 680,
            wl_800 = 800) -> np.ndarray:
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_680: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers
                
        Return 
        ------
            np.ndarray        
    """
    
    red_band_numbers = get_band_numbers(wl_680, w_data)
    print(red_band_numbers)
    nir_band_numbers = get_band_numbers(wl_800, w_data)
    print(nir_band_numbers)
    red = cube[:, :, red_band_numbers]
    nir = cube[:, :, nir_band_numbers]

    mask = (nir - red) / (nir + red) + 1
    mask[nir + red == 0] = 0

    return normalization(mask)


def dvi_mask(cube: np.ndarray,
            w_data: list,
            wl_700=700,
            wl_800=800,
            ) -> np.ndarray:
    """
    dvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_700: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers

        Return 
        ------
            np.ndarray        
    """
    
    band_numbers_700 = get_band_numbers(wl_700,  w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_700 = cube[:, :, band_numbers_700]
    channel_800 = cube[:, :, band_numbers_800]

    mask = channel_800 - channel_700

    return normalization(mask)


def osavi_mask(cube: np.ndarray,
            w_data: list,
            wl_670=670,
            wl_800=800) -> np.ndarray:
    """
    osavi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_670: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_670 = cube[:, :, band_numbers_670]
    channel_800 = cube[:, :, band_numbers_800]
    
    mask = 1.16 * (channel_800 - channel_670) / (channel_800 + channel_670 + 0.16)
    mask[channel_800 + channel_670 + 0.16 == 0] = 0

    return normalization(mask)


def sr_mask(cube: np.ndarray,
            w_data: list,
            wl_680=680,
            wl_800=800
            ) -> np.ndarray:
    """
    sr_mask(cube: np.ndarray, w_data: list)
    
        Calculating the SR index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_680: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers

        Return 
        ------
            np.ndarray        
    """
    
    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube[:, :, band_numbers_680]
    channel_800 = cube[:, :, band_numbers_800]
    
    mask = channel_800 / channel_680
    mask[channel_680 == 0] = 0

    return normalization(mask)


def wdrvi_mask(cube: np.ndarray,
            w_data: list,
            wl_680=680,
            wl_800=800
            ) -> np.ndarray:
    """
    wdrvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the wdrvi index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_680: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube[:, :, band_numbers_680]
    channel_800 = cube[:, :, band_numbers_800]

    mask = (0.05 * channel_800 - channel_680)/(0.05 * channel_800 + channel_680)
    mask[0.05 * channel_800 + channel_680 == 0] = 0

    return normalization(mask)


def mtvi2_mask(cube: np.ndarray,
            w_data: list,
            wl_550=550,
            wl_670=670,
            wl_800=800
            ) -> np.ndarray:
    """
    mtvi2_mask(cube: np.ndarray, w_data: list)
    
        Calculating the mtvi2 index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        wl_550: int 
            wavelength in nanometers    

        wl_670: int
            wavelength in nanometers

        wl_800: int 
            wavelength in nanometers

        Return 
        ------
            np.ndarray        
    """

    band_numbers_550 = get_band_numbers(wl_550, w_data)
    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_550 = cube[:, :, band_numbers_550]
    channel_670 = cube[:, :, band_numbers_670]
    channel_800 = cube[:, :, band_numbers_800]

   
    
    a = 1.5 * (1.2 * (channel_800 - channel_550) - 2.5 * (channel_670 - channel_550))
    b = np.sqrt((2 * channel_800 + 1)**2 - (6 * channel_800 - 5 * np.sqrt(channel_670)) - 0.5)
    mask = a * b
    #При получении nan ставим 0
    mask[np.isnan(mask)==True] = 0
   

    return normalization(mask)