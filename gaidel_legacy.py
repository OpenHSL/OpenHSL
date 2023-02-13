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
CAMERA_PITCH = 0.0
SPECTRUM_BOX_COLOR = (255, 0, 0)
BLUR_AUTO = True
TARGET_RESOLUTION = 1080
CAMERA_TANGENT = 0.30
DISTANCE_LIMIT = 1.0
BLUR_SHAPE = (3, 3)
COS_PITCH = math.cos(math.radians(CAMERA_PITCH))
TAN_PITCH = math.tan(math.radians(CAMERA_PITCH))
PI_2 = math.pi / 2.0

def build_hypercube_by_videos(cube: np.ndarray, gps_filename: str) -> np.ndarray:
    x, y, z = cube.shape #in gaidel_legacy it's x=m, y=k, z=n
    gps = pd.read_csv(gps_filename, delimiter=CSV_DELIMITER)
    gps = gps.loc[gps[HEADER_CAM_ID] == GPS_HYPERCAM_FRAME].head(x)

    bands = interpolate(cube,
                        latitude=gps[HEADER_X].tolist(),
                        longitude=gps[HEADER_Y].tolist(),
                        rel_alt=gps[HEADER_REL_ALT].tolist(),
                        angle=gps[HEADER_ANGLE].tolist())

    blur_band = lambda band: blur_image(band) if BLUR_AUTO else band
    bands = list(map(blur_band, [bands[:, :, i] for i in range(z)]))
    bands = np.array(bands)
    bands = np.transpose(bands, (2, 1, 0))
    print(bands.shape)
    return bands
#-------------------------------------------------------------------------------------------------------------------------------

def interpolate(cube: np.ndarray, latitude: list, longitude: list, rel_alt: list, angle: list) -> np.ndarray:
    n, _, k = cube.shape
    rel_alt = calculate_rel_alt(rel_alt, n)
    latitude, longitude = calculate_lat_lon(latitude, longitude, rel_alt, angle, n)
    x, y, z = coordinates_for_frame(cube, latitude, longitude, rel_alt, angle)
    model = knn_for_interpolate(x, y, z)
    test, n_target, m_target = generate_test_points(x, y)
    prediction = model.predict(test)
    nearest_dit, _ = model.kneighbors(test, n_neighbors=1, return_distance=True)

    #check if the nearest distance is greater than the limit
    for i in range(prediction.shape[0]):
        if nearest_dit[i, 0] > DISTANCE_LIMIT:
            prediction[i, :] = np.zeros(k)

    prediction = prediction.reshape(m_target, n_target, k)
    prediction = np.flip(prediction, axis=2) #####experiment here
    return prediction
#-------------------------------------------------------------------------------------------------------------------------------

def calculate_rel_alt(rel_alt: list, x: int) -> list:
    for j in range(x):
        rel_alt[j] /= COS_PITCH
    return rel_alt

#-------------------------------------------------------------------------------------------------------------------------------

def calculate_lat_lon(latitude: list, longitude: list, rel_alt: list, angle: list, x: int):
    for j in range(x):
        latitude[j] += rel_alt[j] * TAN_PITCH * math.cos(angle[j])
        longitude[j] += rel_alt[j] * TAN_PITCH * math.sin(angle[j])
    return (latitude, longitude)

# ----------------------------------------------------------------------------------------------------------------------

def coordinates_for_frame(cube: np.ndarray, latitude: list, longitude: list, rel_alt: list, angle: list):
    n, m, k = cube.shape
    x = []
    y = []
    z = []
    for j in range(n):
        x_1, y_1 = move_point(latitude[j], longitude[j], angle[j] + PI_2, rel_alt[j] * CAMERA_TANGENT / 2.0)
        x_2, y_2 = move_point(latitude[j], longitude[j], angle[j] - PI_2, rel_alt[j] * CAMERA_TANGENT / 2.0)
        x.extend(np.linspace(x_1, x_2, m))
        y.extend(np.linspace(y_1, y_2, m))
        for index in range(m):
            z.append(np.flip(cube[j,index,:]))
    return (x, y, z)
#-------------------------------------------------------------------------------------------------------------------------------

def move_point(latitude: float, longitude: float, angle: float, distance: float):
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return (latitude + distance * cos_a, longitude + distance * sin_a)

# ----------------------------------------------------------------------------------------------------------------------

def knn_for_interpolate(x, y, z):
    model = neighbors.KNeighborsRegressor(n_neighbors=1)
    data = np.stack((np.array(x), np.array(y))).T #########here is the problem
    model.fit(data, z)
    return model

# ----------------------------------------------------------------------------------------------------------------------

def generate_test_points(x, y):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    n_target = TARGET_RESOLUTION
    m_target = int(n_target * (y_max - y_min) / (x_max - x_min))
    test_points = list(itertools.product(np.linspace(x_min, x_max, n_target), np.linspace(y_min, y_max, m_target)))
    return (test_points, m_target, n_target) ##########here is the problem

# ----------------------------------------------------------------------------------------------------------------------
def blur_image(img: np.ndarray) -> np.ndarray:
    return cv2.blur(img, BLUR_SHAPE)