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
COS_PITCH = math.cos(math.radians(CAMERA_PITCH))
TAN_PITCH = math.tan(math.radians(CAMERA_PITCH))
PI_2 = math.pi / 2.0

def gaussian(length: int, mean: float, std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
# ----------------------------------------------------------------------------------------------------------------------

def blur_image(img: np.ndarray) -> np.ndarray:
    return cv2.blur(img, BLUR_SHAPE)
# ----------------------------------------------------------------------------------------------------------------------

def get_principal_slices(spectrum: np.ndarray) -> np.ndarray:
    """
    # TODO Add description
    Args:
        spectrum: np.ndarray with shape (height, width, 1)

    Returns:
        np.ndarray with shape (BANDS_NUMBER, width, 1)

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
    # create empty array with shape (BANDS_NUMBER, width of frame, 1)

    for j in range(BANDS_NUMBER):
        left_bound = j * n // BANDS_NUMBER

        ans[j, :, :] = np.tensordot(spectrum[left_bound:left_bound + len(gaussian_window), :, :],
                                    gaussian_window,
                                    axes=([0], [0]),)
    return ans
# ----------------------------------------------------------------------------------------------------------------------

def save_slices(data: np.ndarray) -> np.ndarray:
    _, height, _ = data.shape
    cube = np.concatenate([get_principal_slices(frame[int(height * BORDER_TOP):int(height * BORDER_BOTTOM), :, np.newaxis])[:, :, 0][:, np.newaxis] for frame in data], axis=1)
    return cube
# ----------------------------------------------------------------------------------------------------------------------

def build_by_gps_log(spectrum: np.ndarray, gps_filename: str) -> np.ndarray:
    n, m, _ = spectrum.shape
    gps = pd.read_csv(gps_filename, sep=CSV_DELIMITER)
    gps = gps.loc[gps[HEADER_CAM_ID] == GPS_HYPERCAM_FRAME].head(m)

    bands = interp(spectrum,
                   latitude=gps[HEADER_X].tolist(),
                   longitude=gps[HEADER_Y].tolist(),
                   rel_alt=gps[HEADER_REL_ALT].tolist(),
                   angle=gps[HEADER_ANGLE].tolist(),)
    
    blur_band = lambda band: blur_image(band) if BLUR_AUTO else band
    bands = list(map(blur_band, [bands[i, :, :] for i in range(n)]))
    bands = np.array(bands)

    return np.array(bands)
# ----------------------------------------------------------------------------------------------------------------------

def move_point(latitude: float, longitude: float, angle: float, length: float):
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return (
        latitude + length * cos_a,
        longitude + length * sin_a,)
# ----------------------------------------------------------------------------------------------------------------------
def calculate_rel_alt(rel_alt: list, m: int) -> list:
    for j in range(m):
        rel_alt[j] /= COS_PITCH
    return rel_alt

def calculate_lat_lon(latitude: list, longitude: list, rel_alt: list, angle: list, m: int):
    for j in range(m):
        latitude[j] += rel_alt[j] * TAN_PITCH * math.cos(angle[j])
        longitude[j] += rel_alt[j] * TAN_PITCH * math.sin(angle[j])
    return (latitude, longitude)

def interp(lines: np.ndarray, latitude: list, longitude: list, rel_alt: list, angle: list) -> np.ndarray:
    """
    
    """
    n, m, k = lines.shape
    rel_alt = calculate_rel_alt(rel_alt, m)
    latitude, longitude = calculate_lat_lon(latitude, longitude, rel_alt, angle, m)
    x, y, z = calculate_coordinates(latitude, longitude, rel_alt, angle, k, lines, m)
    model = train_model(x, y, z)
    test, n_target, m_target = generate_test_points(x, y)
    prediction = model.predict(test)
    nearest_dist, _ = model.kneighbors(test, n_neighbors=1, return_distance=True)

    # Check if the nearest point is greater than the limit
    for i in range(prediction.shape[0]):
        if nearest_dist[i, 0] > DISTANCE_LIMIT:
            prediction[i, :] = np.zeros(n)

    # Reshape and rearrange the prediction array
    prediction = prediction.reshape(n_target, m_target, n)
    prediction = np.flip(prediction, axis=2)
    prediction = np.transpose(prediction, (2, 1, 0))
    return prediction

def calculate_coordinates(latitude: list, longitude: list, rel_alt: list, angle: list, k: int, lines: np.ndarray, m: int):
    x = []
    y = []
    z = []
    for j in range(m):
        x_1, y_1 = move_point(latitude[j], longitude[j], angle[j] + PI_2, rel_alt[j] * CAMERA_TANGENT / 2.0)
        x_2, y_2 = move_point(latitude[j], longitude[j], angle[j] - PI_2, rel_alt[j] * CAMERA_TANGENT / 2.0)
        x.extend(np.linspace(x_1, x_2, k))
        y.extend(np.linspace(y_1, y_2, k))
        for index in range(k):
            z.append(np.flip(lines[:, j, index]))
    return (x, y, z)

def train_model(x, y, z):
    model = neighbors.KNeighborsRegressor(n_neighbors=1)
    data = np.stack((np.array(x), np.array(y))).T
    model.fit(data, z)
    return model

def generate_test_points(x, y):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    n_target = TARGET_RESOLUTION
    m_target = int(n_target * (y_max - y_min) / (x_max - x_min))
    test_points = list(itertools.product(np.linspace(x_min, x_max, n_target), np.linspace(y_min, y_max, m_target)))
    return (test_points, n_target, m_target)

#----------------------------------------------------------------------------------------------------------------------

def build_hypercube_by_videos(data: np.ndarray, gps_filename: str) -> np.ndarray:
    cube = save_slices(data)
    cube = build_by_gps_log(cube, gps_filename)
    return cube
# ----------------------------------------------------------------------------------------------------------------------
