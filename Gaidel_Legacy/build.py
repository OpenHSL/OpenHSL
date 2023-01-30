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

def build_by_gps_log(spectrum, gps_filename):
    n, m, _ = spectrum.shape
    gps = pd.read_csv(gps_filename, sep=settings.CSV_DELIMITER)
    gps = gps.loc[gps[settings.HEADER_CAM_ID] == settings.GPS_HYPERCAM_FRAME].head(m)
    bands = interp(
        spectrum,
        latitude=gps[settings.HEADER_X].tolist(),
        longitude=gps[settings.HEADER_Y].tolist(),
        rel_alt=gps[settings.HEADER_REL_ALT].tolist(),
        angle=gps[settings.HEADER_ANGLE].tolist(),
    )
    for i in range(n):
        band = bands[i, :, :]
        settings.BLUR_AUTO = True
        if settings.BLUR_AUTO:
            band = blur_image(band)
        bands[i, :, :] = band
    return np.array(bands)

def save_slices(path, return_cube=False):
    capture = cv2.VideoCapture(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    deep = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ans = np.zeros(shape=(settings.config.spectral_bands_number, deep, width), dtype=np.uint8)
    index = 0
    while capture.isOpened():
        result, frame = capture.read()
        if not result:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame)

        s, _ = slices.get_principal_slices(frame[int(height * settings.BORDER_TOP):int(height * settings.BORDER_BOTTOM), :, np.newaxis])
        ans[:, index, :] = s[:, :, 0]
        if cv2.waitKey(1) & 0xFF == ord(settings.EXIT_KEY):
            break
        index += 1
    capture.release()

    if return_cube:
        return ans

def move_point(latitude, longitude, angle, length):
    len_latitude = length * math.cos(angle)
    len_longitude = length * math.sin(angle)
    return (
        latitude + len_latitude,
        longitude + len_longitude,
    )

def interp(lines, latitude, longitude, rel_alt, angle):
    n, m, k = lines.shape
    x = []
    y = []
    z = []
    for j in range(m):
        rel_alt[j] /= math.cos(math.radians(settings.CAMERA_PITCH))
        w = rel_alt[j] * settings.CAMERA_TANGENT
        latitude[j] += rel_alt[j] * math.tan(math.radians(settings.CAMERA_PITCH)) * math.cos(angle[j])
        longitude[j] += rel_alt[j] * math.tan(math.radians(settings.CAMERA_PITCH)) * math.sin(angle[j])
        x_1, y_1 = move_point(latitude[j], longitude[j], angle[j] + math.pi / 2.0, w / 2.0)
        x_2, y_2 = move_point(latitude[j], longitude[j], angle[j] - math.pi / 2.0, w / 2.0)
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


def build_hypercube_by_videos(dir_input, gps_filename):
    cubes = []
    for filename in load_data(dir_input, ".avi"):
        print(filename)
        cube = save_slices(filename, return_cube=True)
        cubes.append(cube)
    cube = np.concatenate(cubes, axis=1)
    cube = build_by_gps_log(cube, gps_filename)
    return cube