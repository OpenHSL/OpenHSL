import math
import numpy as np
import Gaidel_Legacy.settings as settings

def gaussian(length, mean, std):
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std

def get_principal_slices(spectrum: np.ndarray) -> np.ndarray:
    n, m, k = spectrum.shape 
    # n: height of frame, m: width of frame, k = 1

    width = n // settings.config.spectral_bands_number
    gaussian_window = gaussian(width, width / 2.0, width / 6.0) 
    # gaussian_window contain few float data
    
    mid = len(gaussian_window) // 2 
    # center idx of gaussian_window list
    
    gaussian_window[mid] = 1.0 - gaussian_window[:mid].sum() - gaussian_window[mid+1:].sum()

    ans = np.zeros((settings.config.spectral_bands_number, m, k), dtype=np.uint8)
    # create empty array with shape (40, width of frame, 1)

    for j in range(settings.config.spectral_bands_number):
        left_bound = j * n // settings.config.spectral_bands_number

        ans[j, :, :] = np.tensordot(spectrum[left_bound:left_bound + len(gaussian_window), :, :],
                                    gaussian_window,
                                    axes=([0], [0]),)

    return ans
