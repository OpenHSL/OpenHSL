import cv2
import math
import numpy as np
import pandas as pd
import itertools
from sklearn import neighbors

CSV_DELIMITER = ';'
GPS_HYPERCAM_FRAME = "Hypercam frame"
HEADER_CAM_ID = "cam_ID"
HEADER_X = "x"
HEADER_Y = "y"
HEADER_REL_ALT = "rel_alt"
HEADER_ANGLE = "compass_hdg"
BORDER_TOP = 0.0
BORDER_BOTTOM = 1.0
CAMERA_PITCH = 0.0
SPECTRUM_BOX_COLOR = (255, 0, 0)
BLUR_AUTO = True
TARGET_RESOLUTION = 1080
CAMERA_TANGENT = 0.30
DISTANCE_LIMIT = 1.0
BANDS_NUMBER = 40
BLUR_SHAPE = (3, 3)


def gaussian(length, mean, std):
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
# ----------------------------------------------------------------------------------------------------------------------


def get_principal_slices(spectrum: np.ndarray) -> np.ndarray:
    """
    # TODO create doctring here!
    Args:
        spectrum:

    Returns:

    """
    n, m, k = spectrum.shape 
    # n: height of frame, m: width of frame, k = 1

    width = n // BANDS_NUMBER
    gaussian_window = gaussian(width, width / 2.0, width / 6.0) 
    # gaussian_window contain few float data
    
    mid = len(gaussian_window) // 2 
    # center idx of gaussian_window list
    
    gaussian_window[mid] = 1.0 - gaussian_window[:mid].sum() - gaussian_window[mid+1:].sum()

    ans = np.zeros((BANDS_NUMBER, m, k), dtype=np.uint8)
    # create empty array with shape (40, width of frame, 1)

    for j in range(BANDS_NUMBER):
        left_bound = j * n // BANDS_NUMBER

        ans[j, :, :] = np.tensordot(spectrum[left_bound:left_bound + len(gaussian_window), :, :],
                                    gaussian_window,
                                    axes=([0], [0]),)
    return ans
# ----------------------------------------------------------------------------------------------------------------------


def blur_image(img):
    return cv2.blur(img, BLUR_SHAPE)
# ----------------------------------------------------------------------------------------------------------------------


def build_by_gps_log(spectrum: np.ndarray, gps_filename: str):
    n, m, _ = spectrum.shape
    gps = pd.read_csv(gps_filename, sep=CSV_DELIMITER)
    gps = gps.loc[gps[HEADER_CAM_ID] == GPS_HYPERCAM_FRAME].head(m)

    bands = interp(spectrum,
                   latitude=gps[HEADER_X].tolist(),
                   longitude=gps[HEADER_Y].tolist(),
                   rel_alt=gps[HEADER_REL_ALT].tolist(),
                   angle=gps[HEADER_ANGLE].tolist(),)
    # TODO looks like not good, maybe replace by some lambda-function?
    for i in range(n):
        band = bands[i, :, :]
        if BLUR_AUTO:
            band = blur_image(band)
        bands[i, :, :] = band

    return np.array(bands)
# ----------------------------------------------------------------------------------------------------------------------


# TODO This code makes some sense except getting get_principal_slices for each layer?
def save_slices(data: np.ndarray) -> np.ndarray:
    deep, height, width = data.shape
    # create cube with shape (40, deep, width)
    cube: np.ndarray = np.zeros(shape=(BANDS_NUMBER, deep, width), dtype=np.uint8)
    # Todo use enumerate() instead of index +=1
    index = 0 # index in this code also means frame counter
    for frame in data:
        # in get_principal_slices get np.ndarray with shape (height, width, 1), i guess it's specter
        s: np.ndarray = get_principal_slices(frame[int(height * BORDER_TOP):int(height * BORDER_BOTTOM), :, np.newaxis])
        # in s return array with the same shape (height, width, 1)
        cube[:, index, :] = s[:, :, 0]
        index += 1
    return cube
# ----------------------------------------------------------------------------------------------------------------------


def move_point(latitude, longitude, angle, length):
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return (
        latitude + length * cos_a,
        longitude + length * sin_a,
    )
# ----------------------------------------------------------------------------------------------------------------------


# TODO can be split to any functions? Because here is too lot of code.
def interp(lines: np.ndarray, latitude: list, longitude: list, rel_alt: list, angle: list) -> np.ndarray:
    """
        This code implements interpolation of 3D uav data.
        The input data is a 3D array of shape (n, m, k), where n is the number of spectral bands,
        m is the number of points in the flight path, and k is the number of points along the line.
        
        The function takes 4 additional arguments:
        latitude, longitude, rel_alt, angle - lists of length m, which contain the coordinates of the points
        of the flight path, the relative altitude and the angle of the camera at each point.
        The code transforms the coordinates of each line and maps the data to a 2D plane.
        This is done by looping over each line and first adjusting the atlitude by diving rel_alt by the cosine of the camera pitch.
        Then, the latitude and longitude are shifted using the adjusted altitude, angle. and camera pitch.
    """
    n, m, k = lines.shape
    x = []
    y = []
    z = []
    cos_pitch = math.cos(math.radians(CAMERA_PITCH))
    tan_pitch = math.tan(math.radians(CAMERA_PITCH))
    pi_2 = math.pi / 2.0
    for j in range(m):
        rel_alt[j] /= cos_pitch
        w = rel_alt[j] * CAMERA_TANGENT
        latitude[j] += rel_alt[j] * tan_pitch * math.cos(angle[j])
        longitude[j] += rel_alt[j] * tan_pitch * math.sin(angle[j])
        x_1, y_1 = move_point(latitude[j], longitude[j], angle[j] + pi_2, w / 2.0)
        x_2, y_2 = move_point(latitude[j], longitude[j], angle[j] - pi_2, w / 2.0)
        x.extend(np.linspace(x_1, x_2, k))
        y.extend(np.linspace(y_1, y_2, k))
        for index in range(k):
            z.append(np.flip(lines[:, j, index]))

    model = neighbors.KNeighborsRegressor(n_neighbors=1)
    data = np.stack((np.array(x), np.array(y))).T
    model.fit(data, z)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    n_target = TARGET_RESOLUTION
    m_target = int(n_target * (y_max - y_min) / (x_max - x_min))
    test = list(itertools.product(np.linspace(x_min, x_max, n_target), np.linspace(y_min, y_max, m_target)))
    ans = model.predict(test)
    neigh_dist, _ = model.kneighbors(test, n_neighbors=1, return_distance=True)
    for i in range(ans.shape[0]):
        if neigh_dist[i, 0] > DISTANCE_LIMIT:
            ans[i, :] = np.zeros(n)
    ans = ans.reshape(n_target, m_target, n)
    ans = np.flip(ans, axis=2)
    ans = np.transpose(ans, (2, 1, 0))
    return ans
# ----------------------------------------------------------------------------------------------------------------------


def build_hypercube_by_videos(data: np.ndarray, gps_filename: str) -> np.ndarray:
    cube = save_slices(data)
    cube = build_by_gps_log(cube, gps_filename)
    return cube
# ----------------------------------------------------------------------------------------------------------------------
