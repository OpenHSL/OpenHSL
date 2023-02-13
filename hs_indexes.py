import numpy as np

# TODO which type is w_data?
# TODO rewrite docstring
def ndvi_mask(cube: np.ndarray, w_data: str):
    """
    ____________
    necessary to read three-dimensional images and a text file with wavelengths
    ____________
    Parameters
    ----------
    cube : np.ndarray, 
    w_dana : str
            """
    # TODO bad naming
    def sps(x, y, w_data):

        # The input is supplied with wavelengths in the form of a list
        wd = [i for i, v in enumerate(w_data) if x < v < y]
        # TODO bad naming
        sps = [min(wd), max(wd)] 
        return sps

    # TODO bad naming
    def reduceMean(HSI, l_bound, r_bound):
        return np.mean(HSI[:, :, l_bound:r_bound], axis=2)

    # TODO replace code in ndvi_mask
    def NDVI(cube):

        # TODO bad naming
        r1 = 633
        # TODO bad naming
        r2 = 650
        # TODO bad naming
        red_sps = sps(r1, r2, w_data)

        # TODO bad naming
        n1 = 844
        # TODO bad naming
        n2 = 860
        # TODO bad naming
        nir_sps = sps(n1, n2, w_data)

        red = reduceMean(cube, red_sps[0], red_sps[1])
        nir = reduceMean(cube, nir_sps[0], nir_sps[1])

        # TODO normalize values in [0, 1]
        # TODO add zero-replacement elements where nir + red = 0.0
        # TODO delete + 0.0001
        return (nir - red)/(nir + red+0.0001)
    
    mask = NDVI(cube)
    return mask
