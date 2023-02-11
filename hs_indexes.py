import numpy as np

def ndvi_mask(cube : np.ndarray, w_data : str):
    """
    ____________
    necessary to read three-dimensional images and a text file with wavelengths
    ____________
    Parameters
    ----------
    cube : np.ndarray, 
    w_dana : str
            """
    def sps(x, y, w_data):

        # The input is supplied with wavelengths in the form of a list
        wd = [i for i, v in enumerate(w_data) if x < v < y]
        sps = [min(wd), max(wd)] 
        return sps

    def reduceMean(HSI, l_bound, r_bound):

        return np.mean(HSI[:, :, l_bound:r_bound], axis=2)

    def NDVI(cube):

        r1 = 633
        r2 = 650
        red_sps = sps(r1, r2, w_data)
        

        n1 = 844
        n2 = 860
        nir_sps = sps(n1, n2, w_data)
        

        red = reduceMean(cube, red_sps[0], red_sps[1])
        nir = reduceMean(cube, nir_sps[0], nir_sps[1])
       
        return (nir - red)/(nir + red+0.0001)
    
    mask = NDVI(cube)
    return mask 