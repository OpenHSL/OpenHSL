import numpy as np

from itertools import product
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import trange

from openhsl.base.hsi import HSImage


def __get_cluster(cl_type):
    if cl_type == 'KMeans':
        return KMeans
    elif cl_type == 'SpectralClustering':
        return SpectralClustering
# ----------------------------------------------------------------------------------------------------------------------


def cluster_hsi(hsi: HSImage,
                n_clusters: int = 2,
                cl_type='Kmeans') -> np.ndarray:
    km = __get_cluster(cl_type=cl_type)(n_clusters=n_clusters)
    h, w, _ = hsi.data.shape
    pred = km.fit_predict(hsi.to_spectral_list())
    return pred.reshape((h, w))
# ----------------------------------------------------------------------------------------------------------------------


def ANDVI(hsi: HSImage):
    def ndi(img: np.ndarray,
            l_red: int,
            r_red: int,
            l_nir: int,
            r_nir: int):
        red = np.mean(img[:, :, l_red: r_red], axis=2)
        nir = np.mean(img[:, :, l_nir: r_nir], axis=2)
        ndi_mask = (nir - red) / (nir + red)
        ndi_mask[nir + red == 0] = 0
        return ndi_mask > 0.1

    def get_ttest(img: np.ndarray,
                  plant_mask: np.ndarray):
        plant = img[plant_mask == 1]
        soil = img[plant_mask == 0]
        return ttest_ind(plant[: 150], soil[: 150]) #

    p_v = []
    for i in trange(98, 148):
        mask = ndi(hsi.data, 97, i, 148, 250)
        p_v.append(np.mean(get_ttest(hsi.data, mask)[1]))
    res_red_right = int(np.argmin(np.log(p_v)))
    print(f'right border of red is {res_red_right + 97}\'s band')
    return ndi(hsi.data, 97, res_red_right + 97, 148, 250)
# ----------------------------------------------------------------------------------------------------------------------


def norm_diff_index(channel_1: np.ndarray,
                    channel_2: np.ndarray) -> np.ndarray:
    mask = (channel_1 - channel_2) / (channel_1 + channel_2)
    mask[np.isnan(mask)] = 1
    magic_threshold = 2
    mask[mask < magic_threshold] = 0
    mask[mask >= magic_threshold] = 1
    return mask
# ----------------------------------------------------------------------------------------------------------------------


def ANDI(hsi: HSImage,
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

    ndi = norm_diff_index(channel_1=hsi.data[:, :, best_idx[0]],
                          channel_2=hsi.data[:, :, best_idx[1]])

    return ndi
# ----------------------------------------------------------------------------------------------------------------------