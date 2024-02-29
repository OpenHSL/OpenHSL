import numpy as np
from itertools import product
from typing import Union, Tuple
from openhsl.hs_mask import HSMask
from openhsl.hsi import HSImage
from scipy.interpolate import interp1d


def normalization(mask: np.ndarray) -> np.ndarray:
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


def neighbor_el(elements_list: list, element: float) -> float:
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


def get_band_numbers(w_l: int, w_data: Union[list, np.ndarray]) -> int:
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


def get_hypercube_and_wavelength(cube: Union[HSImage, np.ndarray],
                                 wave_data: Union[list, np.ndarray] = None) -> Tuple[np.ndarray, list]:
    """
    get_hypercube_and_wavelength(cube, wave_data)

        Returns hypercube and wavelengths, determines priority

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            np.ndarray, list
    """

    w_data = None

    if isinstance(cube, HSImage):
        cube_data = cube.data
        if cube.wavelengths:
            w_data = cube.wavelengths

    elif isinstance(cube, np.ndarray):
        cube_data = cube
    else:
        raise ValueError("Unvailable type of HSI")

    if np.any(wave_data):
        w_data = list(wave_data)

    if not w_data:
        raise ValueError("Not info about wavelengths")

    if type(w_data) != list:
        w_data = list(w_data)

    return cube_data, w_data


def ndvi_mask(cube: Union[HSImage, np.ndarray],
              wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    ndvi_mask(cube, wave_data)

        Calculating the NDVI index -
        "https://www.bestdroneconsulting.com/support/knowledge-base/97-vegetation-indices-ndvi-and-dvi.html,
        https://gis-lab.info/qa/ndvi.html, https://gis-lab.info/qa/vi.html,
        https://iopscience.iop.org/article/10.1088/1742-6596/1003/1/012083/pdf"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    print('w_data - ', w_data)

    wl_680 = 640
    wl_800 = 850

    red_band_numbers = get_band_numbers(wl_680, w_data)
    nir_band_numbers = get_band_numbers(wl_800, w_data)

    print(red_band_numbers, nir_band_numbers)

    red = cube_data[:, :, red_band_numbers].astype(float)
    nir = cube_data[:, :, nir_band_numbers].astype(float)

    mask = (nir - red) / (nir + red)
    mask[nir + red == 0] = 0

    # index_mask = HSMask(mask, None)

    return mask


def wbi_mask(cube: Union[HSImage, np.ndarray],
             wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    wbi_mask(cube, wave_data)
        Calculating the wbi index -
        https://gisproxima.ru/ispolzovanie_vegetatsionnyh_indeksov#:~:text=%D0%92%D0%BE%D0%B4%D0%BD%D1%8B%D0%B9%20%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%20(WBI),
        %D0%BF%D0%BE%D0%B3%D0%BB%D0%BE%D1%89%D0%B5%D0%BD%D0%B8%D0%B5%D0%BC%20%D0%B2%20%D0%B4%D0%B8%D0%B0%D0%BF%D0%B0%D0%B7%D0%BE%D0%BD%D0%B5%20900%20%D0%BD%D0%BC.

        https://sovzond.ru/upload/iblock/f46/2011_02_017.pdf


        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    print('w_data - ', w_data)

    wl_900 = 900
    wl_970 = 970

    nir_900_band_numbers = get_band_numbers(wl_900, w_data)
    nir_970_band_numbers = get_band_numbers(wl_970, w_data)

    print(nir_900_band_numbers, nir_970_band_numbers)

    nir_900 = cube_data[:, :, nir_900_band_numbers].astype(float)
    nir_970 = cube_data[:, :, nir_970_band_numbers].astype(float)

    mask = nir_900 / nir_970
    mask[nir_970 == 0] = 0

    # index_mask = HSMask(mask, None)

    return mask


def ndwi_mask(cube: Union[HSImage, np.ndarray],
              wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    ndwi_mask(cube, wave_data)

        Calculating the NDWI index - https://earchive.tpu.ru/bitstream/11683/67170/1/TPU1166721.pdf


        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    print('w_data - ', w_data)

    wl_680 = 559
    wl_800 = 864

    red_band_numbers = get_band_numbers(wl_680, w_data)
    nir_band_numbers = get_band_numbers(wl_800, w_data)

    print(red_band_numbers, nir_band_numbers)

    red = cube_data[:, :, red_band_numbers].astype(float)
    nir = cube_data[:, :, nir_band_numbers].astype(float)

    mask = (red - nir) / (nir + red)
    mask[nir + red == 0] = 0

    # index_mask = HSMask(mask, None)

    return mask


def dvi_mask(cube: Union[HSImage, np.ndarray],
             wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    dvi_mask(cube, wave_data)

        Calculating the DVI index -
        "https://www.bestdroneconsulting.com/support/knowledge-base/97-vegetation-indices-ndvi-and-dvi.html,
        https://iopscience.iop.org/article/10.1088/1742-6596/1003/1/012083/pdf"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_700 = 700
    wl_800 = 800

    band_numbers_700 = get_band_numbers(wl_700, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    print(band_numbers_700, band_numbers_800)

    channel_700 = cube_data[:, :, band_numbers_700].astype(float)
    channel_800 = cube_data[:, :, band_numbers_800].astype(float)

    mask = channel_800 - channel_700
    # index_mask = HSMask(mask, None)

    return mask


def osavi_mask(cube: Union[HSImage, np.ndarray],
               wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    osavi_mask(cube, wave_data)

        Calculating the 'osavi' index -
        "https://support.micasense.com/hc/en-us/articles/227837307-Overview-of-Agricultural-Indices#osavi,
        https://www.sciencedirect.com/science/article/pii/S0034425797001144"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_670 = 670
    wl_800 = 800

    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_670 = cube_data[:, :, band_numbers_670].astype(float)
    channel_800 = cube_data[:, :, band_numbers_800].astype(float)

    mask = 1.16 * (channel_800 - channel_670) / (channel_800 + channel_670 + 0.16)
    mask[channel_800 + channel_670 + 0.16 == 0] = 0

    # index_mask = HSMask(mask, None)

    return mask


def norm_diff_index(channel_1: np.ndarray,
                    channel_2: np.ndarray) -> np.ndarray:
    mask = (channel_1 - channel_2) / (channel_1 + channel_2)
    mask[np.isnan(mask)] = 1
    magic_threshold = 2
    mask[mask < magic_threshold] = 0
    mask[mask >= magic_threshold] = 1
    return mask


def adaptive_norm_diff_index(cube: np.ndarray,
                             example_1: np.ndarray,
                             example_2: np.ndarray) -> np.ndarray:
    example_1_size = example_1[:, :, 0].size
    example_2_size = example_2[:, :, 0].size

    min_score = 2.0
    best_idx = None, None

    if example_1.shape[2] == example_2.shape[2]:
        channels_count = example_1.shape[2]
    else:
        raise ValueError("Different numbers of channels in two sets")

    for ind_1, ind_2 in product(range(channels_count), range(channels_count)):

        # skip main diagonal
        if ind_1 == ind_2:
            continue

        ndi = norm_diff_index(channel_1=example_1[:, :, ind_1],
                              channel_2=example_1[:, :, ind_2])

        score = np.sum(ndi) / example_1_size

        ndi = norm_diff_index(channel_1=example_2[:, :, ind_1],
                              channel_2=example_2[:, :, ind_2])

        score += (example_2_size - np.sum(ndi)) / example_2_size

        if score < min_score:
            min_score = score
            best_idx = ind_1, ind_2

    print(best_idx)

    ndi = norm_diff_index(channel_1=cube[:, :, best_idx[0]],
                          channel_2=cube[:, :, best_idx[1]])

    return ndi


def sr_mask(cube: Union[HSImage, np.ndarray],
            wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    sr_mask(cube, wave_data)

        Calculating the 'SR' index - "https://www.hiphen-plant.com/vegetation-index/3582/"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube_data[:, :, band_numbers_680].astype(float)
    channel_800 = cube_data[:, :, band_numbers_800].astype(float)

    mask = channel_800 / channel_680
    mask[channel_680 == 0] = 0

    # index_mask = HSMask(mask, None)

    return mask


def wdrvi_mask(cube: Union[HSImage, np.ndarray],
               wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    wdrvi_mask(cube, wave_data)

        Calculating the 'wdrvi' index - "https://www.sciencedirect.com/science/article/pii/S0176161704705726"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_680 = 680
    wl_800 = 800

    band_numbers_680 = get_band_numbers(wl_680, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_680 = cube_data[:, :, band_numbers_680].astype(float)
    channel_800 = cube_data[:, :, band_numbers_800].astype(float)

    mask = (0.05 * channel_800 - channel_680) / (0.05 * channel_800 + channel_680)
    mask[(0.05 * channel_800 + channel_680) == 0] = 0

    return mask


def mtvi2_mask(cube: Union[HSImage, np.ndarray],
               wave_data: Union[list, np.ndarray] = None) -> Tuple[HSMask, np.ndarray]:
    """
    mtvi2_mask(cube, wave_data)

        Calculating the 'mtvi2' index - "https://pro.arcgis.com/en/pro-app/latest/arcpy/image-analyst/mtvi2.htm"

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            HSMask, np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_550 = 550
    wl_670 = 670
    wl_800 = 800

    band_numbers_550 = get_band_numbers(wl_550, w_data)
    band_numbers_670 = get_band_numbers(wl_670, w_data)
    band_numbers_800 = get_band_numbers(wl_800, w_data)

    channel_550 = cube_data[:, :, band_numbers_550].astype(float)
    channel_670 = cube_data[:, :, band_numbers_670].astype(float)
    channel_800 = cube_data[:, :, band_numbers_800].astype(float)

    a = 1.5 * (1.2 * (channel_800 - channel_550) - 2.5 * (channel_670 - channel_550))
    b = np.sqrt((2 * channel_800 + 1) ** 2 - (6 * channel_800 - 5 * np.sqrt(channel_670)) - 0.5)
    mask = a * b
    mask[np.isnan(mask)] = 0

    # index_mask = HSMask(mask, None)

    return mask  # index_mask, normalization(mask)


def simple_hsi_to_rgb(cube: Union[HSImage, np.ndarray],
                      wave_data: Union[list, np.ndarray] = None) -> np.ndarray:
    """
    simple_hsi_to_rgb(cube, wave_data)

        Return rgb-image from hyperspectral image

        Parameters
        ----------
        cube: HSImage or np.ndarray
           hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        Returns
        ------
            np.ndarray
    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    wl_440 = 440
    wl_550 = 550
    wl_640 = 640

    blue_band_numbers = get_band_numbers(wl_440, w_data)
    green_band_numbers = get_band_numbers(wl_550, w_data)
    red_band_numbers = get_band_numbers(wl_640, w_data)

    print('BGR band: ', blue_band_numbers, green_band_numbers, red_band_numbers)

    blue = cube_data[:, :, blue_band_numbers].astype(float)
    green = cube_data[:, :, green_band_numbers].astype(float)
    red = cube_data[:, :, red_band_numbers].astype(float)

    return np.dstack((red.astype(np.uint8), green.astype(np.uint8), blue.astype(np.uint8)))


def xyz2srgb_exgamma(xyz: np.ndarray) -> np.ndarray:
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


def hsi_to_rgb(cube: Union[HSImage, np.ndarray],
               wave_data: Union[list, np.ndarray],
               illumination_coef: np.ndarray,
               xyz_bar: np.ndarray) -> np.ndarray:
    """
    hsi_to_rgb(cube, w_data, illumination_coef, xyzbar)

        Extracts an RGB image from an HSI image

        Parameters
        ----------
        cube: HSImage or np.ndarray
            hyperspectral image

        wave_data: list or np.ndarray
            list of hyperspectral images wavelengths

        illumination_coef: np.ndarray
            spectral intensity coefficients of the light source

        xyz_bar: np.ndarray
            color space coefficients CIE 1931

        Returns
        ------
            np.ndarray

    """

    cube_data, w_data = get_hypercube_and_wavelength(cube, wave_data)

    # wavelengths_visible_light_range
    wvlr = list(np.linspace(400, 720, 33))

    right_bound = w_data.index(neighbor_el(w_data, 720))
    left_bound = w_data.index(neighbor_el(w_data, 400))
    rgb_waves = w_data[left_bound:right_bound]

    f_illum = interp1d(wvlr, illumination_coef[:, 0])

    new_cube = np.zeros((np.shape(cube_data)[0], np.shape(cube_data)[1], right_bound))
    for i, w in zip(range(new_cube.shape[-1]), rgb_waves):
        new_cube[:, :, i] = cube_data[:, :, i] * f_illum(w)

    r, c, w = new_cube.shape
    radiances = np.reshape(new_cube, (r * c, w))

    xyz_bar_0 = xyz_bar[:, 0]
    xyz_bar_1 = xyz_bar[:, 1]
    xyz_bar_2 = xyz_bar[:, 2]

    f_0 = interp1d(wvlr, xyz_bar_0)
    f_1 = interp1d(wvlr, xyz_bar_1)
    f_2 = interp1d(wvlr, xyz_bar_2)

    xyz_0 = [f_0(i) for i in rgb_waves]
    xyz_1 = [f_1(i) for i in rgb_waves]
    xyz_2 = [f_2(i) for i in rgb_waves]

    xyz_bar_new = (np.array([xyz_0, xyz_1, xyz_2])).T

    xyz = np.dot(radiances, xyz_bar_new)
    xyz = np.reshape(xyz, (r, c, 3))
    xyz = (xyz - np.min(xyz)) / (np.max(xyz) - np.min(xyz))

    rgb = xyz2srgb_exgamma(xyz)
    rgb = normalization(rgb)

    gray_mean = np.mean(rgb, axis=2)
    un = np.unique(gray_mean)

    thresh = 0.98
    coord = [np.where(gray_mean == un[int(len(un) * thresh - 1)])][0]
    x, y = coord
    m = max(rgb[int(x), int(y), :])

    rgb[rgb > m] = m
    rgb = rgb / np.max(rgb)

    return rgb
