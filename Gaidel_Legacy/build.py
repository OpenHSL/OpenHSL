import cv2
import math
import numpy as np
import pandas as pd
import itertools

import Gaidel_Legacy.settings as settings
from Gaidel_Legacy.utils import load_data
from Gaidel_Legacy.image import blur_image
from sklearn import neighbors

import Gaidel_Legacy.slices as slices


SPECTRUM_BOX_COLOR = (255, 0, 0)

def build_by_gps_log(spectrum: np.ndarray, gps_filename: str):
    n, m, _ = spectrum.shape
    gps = pd.read_csv(gps_filename, sep=settings.CSV_DELIMITER)
    gps = gps.loc[gps[settings.HEADER_CAM_ID] == settings.GPS_HYPERCAM_FRAME].head(m)

    bands = interp(spectrum,
                   latitude=gps[settings.HEADER_X].tolist(),
                   longitude=gps[settings.HEADER_Y].tolist(),
                   rel_alt=gps[settings.HEADER_REL_ALT].tolist(),
                   angle=gps[settings.HEADER_ANGLE].tolist(),)
    
    for i in range(n):
        band = bands[i, :, :]
        settings.BLUR_AUTO = True
        if settings.BLUR_AUTO:
            band = blur_image(band)
        bands[i, :, :] = band

    return np.array(bands)

def save_slices(path: str) -> np.ndarray:
    """
        Args:
            path: path to video file
        Returns:
            cube: ndarray with cube from video
            
    """
    capture = cv2.VideoCapture(path)
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # width of video
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height of video
    deep: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # count of frames in video

    # create cube with shape (40, deep, width)
    cube: np.ndarray = np.zeros(shape=(settings.config.spectral_bands_number, deep, width), dtype=np.uint8)
    index = 0 # index in this code also means frame counter
    while capture.isOpened():
        result, frame = capture.read()
        if not result:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # in get_principal_slices get np.ndarray with shape (height, width, 1), i guess it's specter
        s: np.ndarray = slices.get_principal_slices(frame[int(height * settings.BORDER_TOP):int(height * settings.BORDER_BOTTOM), :, np.newaxis])
        
        # in s return array with the same shape (height, width, 1)

        cube[:, index, :] = s[:, :, 0]
        if cv2.waitKey(1) & 0xFF == ord(settings.EXIT_KEY):
            break
        index += 1
    capture.release()

    return cube

def move_point(latitude, longitude, angle, length):
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return (
        latitude + length * cos_a,
        longitude + length * sin_a,
    )


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
    cos_pitch = math.cos(math.radians(settings.CAMERA_PITCH))
    tan_pitch = math.tan(math.radians(settings.CAMERA_PITCH))
    pi_2 = math.pi / 2.0
    for j in range(m):
        rel_alt[j] /= cos_pitch
        w = rel_alt[j] * settings.CAMERA_TANGENT
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
    n_target = settings.TARGET_RESOLUTION
    m_target = int(n_target * (y_max - y_min) / (x_max - x_min))
    test = list(itertools.product(np.linspace(x_min, x_max, n_target), np.linspace(y_min, y_max, m_target)))
    ans = model.predict(test)
    neigh_dist, _ = model.kneighbors(test, n_neighbors=1, return_distance=True)
    for i in range(ans.shape[0]):
        if neigh_dist[i, 0] > settings.DISTANCE_LIMIT:
            ans[i, :] = np.zeros(n)
    ans = ans.reshape(n_target, m_target, n)
    ans = np.flip(ans, axis=2)
    ans = np.transpose(ans, (2, 1, 0))
    return ans


def build_hypercube_by_videos(dir_input: str, gps_filename: str) -> np.ndarray:
    cubes = []
    for filename in load_data(dir_input, ".avi"): #filename = dir_input + "video_1.avi"
        cube: np.ndarray = save_slices(filename) 
        print(filename)
        cubes.append(cube) # every cube have a shape (40, num of frames, width)
    
    cube = np.concatenate(cubes, axis=1)

    cube = build_by_gps_log(cube, gps_filename)
    return cube