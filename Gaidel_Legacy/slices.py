import math
import numpy
import Gaidel_Legacy.settings as settings

def gaussian(length, mean, std):
    return numpy.exp(-((numpy.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std

def get_principal_slices(spectrum):
    n, m, k = spectrum.shape
    width = n // settings.config.spectral_bands_number
    gaussian_window = gaussian(width, width / 2.0, width / 6.0)
    mid = len(gaussian_window) // 2
    gaussian_window[mid] = 1.0 - numpy.sum(gaussian_window) + gaussian_window[mid]
    ans = numpy.zeros(shape=(settings.config.spectral_bands_number, m, k), dtype=numpy.uint8)
    for j in range(settings.config.spectral_bands_number):
        left_bound = j * n // settings.config.spectral_bands_number
        ans[j, :, :] = numpy.tensordot(
            spectrum[left_bound:left_bound + len(gaussian_window), :, :],
            gaussian_window,
            axes=([0], [0]),
        )
    return ans, numpy.linspace(
            settings.WAVELENGTH_MIN, settings.WAVELENGTH_MAX, settings.config.spectral_bands_number
    ).tolist()
