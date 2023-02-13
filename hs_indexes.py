import numpy as np


def get_band_numbers(left_border: int, right_border: int, w_data: list) -> tuple:
    """
    get_band_numbers(left_border, right_border, w_data)

        The input is supplied with wavelengths in the form of a list
    """
    part_bands_list = [i for i, v in enumerate(w_data) if left_border < v < right_border]
    
    return min(part_bands_list), max(part_bands_list)

   
def reduce_mean(HSI: np.ndarray, l_bound: int, r_bound: int) -> int:
    """
    reduceMean(HSI, l_bound, r_bound):

        Calculating average values from multiple channels
    
    """
    return np.mean(HSI[:, :, l_bound:r_bound], axis=2)


def ndvi_mask(cube: np.ndarray, 
              w_data: list, 
              left_red = 633, 
              right_red = 650, 
              left_nir = 844, 
              right_nir = 860) -> np.ndarray: 
    
    """
    ndvi_mask(cube: np.ndarray, w_data: list)
    
        Calculating the NDVI index
        
        Parameters
        ----------
        cube: np.ndarray
           hyperspectral image

        w_data: list
            list of hyperspectral images wavelengths

        Return 
        ------
            np.ndarray        
        """
    
    red_band_numbers = get_band_numbers(left_red, right_red, w_data)
    nir_band_numbers = get_band_numbers(left_nir, right_nir, w_data)

    red = reduce_mean(cube, red_band_numbers[0], red_band_numbers[1])
    nir = reduce_mean(cube, nir_band_numbers[0], nir_band_numbers[1])

    mask = (nir - red)/(nir + red) + 1
    mask[nir + red == 0] = 0 
    mask_2 = np.copy(mask)
    mask_2 = ((mask_2 - mask_2.min())/(mask_2.max() - mask_2.min()))
    return mask_2