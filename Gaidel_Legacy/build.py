import os
import cv2
import math
import numpy as np
import pandas as pd
import itertools

import Gaidel_Legacy.settings as settings
import Gaidel_Legacy.messages as messages
from Gaidel_Legacy.utils import load_data
from Gaidel_Legacy.image import scale_gray_levels, blur_image
from Gaidel_Legacy.saver import save_slices
from sklearn import neighbors

# -*- coding: utf-8 -*-

import Gaidel_Legacy.saver as saver
import Gaidel_Legacy.slices as slices
import Gaidel_Legacy.split as split
from Gaidel_Legacy.vigntetting import VignettingFixer


SPECTRUM_BOX_COLOR = (255, 0, 0)

def build_by_gps_log(spectrum, gps_filename):
    n, m, k = spectrum.shape  # n - spectral channels, m - spatial lines, k - width of the line
    gps = pd.read_csv(gps_filename, sep=settings.CSV_DELIMITER)
    gps = gps.loc[gps[settings.HEADER_CAM_ID] == settings.GPS_HYPERCAM_FRAME].head(m)
    if settings.VERBOSE:
        print(messages.SPECTRAL_BANDS_BUILDING)
    bands = interp(
        spectrum,
        latitude=gps[settings.HEADER_X].tolist(),
        longitude=gps[settings.HEADER_Y].tolist(),
        rel_alt=gps[settings.HEADER_REL_ALT].tolist(),
        angle=gps[settings.HEADER_ANGLE].tolist(),
    )
    for i in range(n):
        band = bands[i, :, :]
        if settings.config.postprocessing_scaling_gray_levels:
            band = scale_gray_levels(band)
        settings.BLUR_AUTO = True  # DEBUG
        if settings.BLUR_AUTO:
            band = blur_image(band)
        bands[i, :, :] = band
    return np.array(bands)

def show_video(path):
    splitter = split.Splitter()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(messages.CANT_OPEN_PATTERN.format(path))
        return
    while capture.isOpened():
        result, frame = capture.read()
        if not result:
            break
        if settings.DETECT_RAINBOW:
            splitter.split(frame)
            frame = cv2.rectangle(frame, splitter.spectrum_rect[0], splitter.spectrum_rect[1], SPECTRUM_BOX_COLOR)
        cv2.imshow(messages.FRAME_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord(settings.EXIT_KEY):
            break
    capture.release()
    cv2.destroyWindow(messages.FRAME_WINDOW)


def save_slices(path, result_path=None, hyper_format=saver.FORMAT_PNG, _=None, gps_filename=None, return_cube=False):
    if settings.VERBOSE:
        print(messages.SLICES_PROCESSING)
    if result_path is None:
        result_path = os.path.join(settings.PROJECT_DIR, settings.RESULTS_DIR)
    splitter = split.Splitter()
    vignetting_fixer = None
    if settings.config.postprocessing_fix_vignetting:
        vignetting_fixer = VignettingFixer(settings.FILENAME_DATA)

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(messages.CANT_OPEN_PATTERN.format(path))
        return
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    deep = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ans = np.zeros(shape=(settings.config.spectral_bands_number, deep, width), dtype=np.uint8)
    layers = []
    index = 0
    while capture.isOpened():
        result, frame = capture.read()
        if not result:
            break
        if settings.DETECT_RAINBOW:
            splitter.split(frame)
            crop = cv2.cvtColor(frame[
                splitter.spectrum_rect[0][1]:splitter.spectrum_rect[1][1],
                splitter.spectrum_rect[0][0]:splitter.spectrum_rect[1][0]
            ].copy(), cv2.COLOR_RGB2GRAY)
            frame = cv2.rectangle(frame, splitter.spectrum_rect[0], splitter.spectrum_rect[1], SPECTRUM_BOX_COLOR)
            cv2.imshow(messages.FRAME_WINDOW, frame)
            frame = crop
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame)
        if vignetting_fixer is not None:
            frame = vignetting_fixer.fix(frame)
        s, layers = slices.get_principal_slices(
            frame[int(height * settings.BORDER_TOP):int(height * settings.BORDER_BOTTOM), :, np.newaxis]
        )
        ans[:, index, :] = s[:, :, 0]
        if cv2.waitKey(1) & 0xFF == ord(settings.EXIT_KEY):
            break
        index += 1
    capture.release()
    if gps_filename is not None:
        ans = build_by_gps_log(ans, gps_filename)
    if return_cube:
        return ans
    saver.save_slices(
        ans, result_path, os.path.splitext(os.path.basename(path))[0], hyper_format, layers, original_path=path
    )


def save_slices_from_images(path, result_path=None, hyper_format=saver.FORMAT_PNG, _=None, return_cube=False):
    if settings.VERBOSE:
        print(messages.SLICES_PROCESSING)
    if result_path is None:
        result_path = os.path.join(settings.PROJECT_DIR, settings.RESULTS_DIR)
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    if len(files) == 0:
        print(messages.ERROR_DIRECTORY_IS_EMPTY.format(path))
        return
    shape = cv2.imread(files[0]).shape
    width = shape[0]
    height = shape[1]
    deep = len(files)
    ans = np.zeros(shape=(settings.config.spectral_bands_number, deep, width), dtype=np.uint8)
    layers = []
    index = 0
    for name in files:
        frame = cv2.imread(name)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame)
        frame = frame.transpose()
        ans[:, index, :], layers = slices.get_principal_slices(
            frame[np.newaxis, int(height * settings.BORDER_TOP):int(height * settings.BORDER_BOTTOM), :]
        )
        index += 1
    ans = np.transpose(ans, axes=[1, 0, 2])
    if return_cube:
        return ans
    else:
        saver.save_slices(ans, result_path, os.path.splitext(os.path.basename(path))[0], hyper_format, layers)

def move_point(latitude, longitude, angle, length):
    # r = earth_radius(latitude)
    len_latitude = length * math.cos(angle)
    len_longitude = length * math.sin(angle)
    return (
        latitude + len_latitude,
        longitude + len_longitude,
    )

def interp(lines, latitude, longitude, rel_alt, angle):
    n, m, k = lines.shape
    if not settings.VERTICAL_FLIP:
        lines = np.flip(lines, axis=2)
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
    # model = interpolate.interp2d(x, y, z, kind="linear")

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


def build_hypercube_by_videos(dir_input, dir_output, gps_filename, output_format):
    cubes = []
    for filename in load_data(dir_input, ".avi"):
        print(filename)
        cube = save_slices(filename, dir_output, output_format, return_cube=True)
        cubes.append(cube)
    cube = np.concatenate(cubes, axis=1)
    cube = build_by_gps_log(cube, gps_filename)
    print(cube.shape)
    # saver.save_slices(cube, dir_output, os.path.splitext(os.path.basename(gps_filename))[0], output_format)
    return cube