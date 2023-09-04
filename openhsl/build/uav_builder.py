import cv2
import math
import numpy as np
import pandas as pd
import itertools

from sklearn import neighbors
from typing import List, Tuple

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


def blur_band(band):
    return blur_image(band) if BLUR_AUTO else band


def build_hypercube_by_videos(cube: np.ndarray, gps_filename: str) -> np.ndarray:
    """
        build_hypercube_by_videos(cube, gps_filename)

            Load preprocessed frames from UAV to build 3D hypercube by using
            gps telemetry in CSV file

            Parameters
            ----------
            cube: np.ndarray
                Preprocessing array with shape (nums_frames, width, height)
            
            gps_filename: str
                Contain telemetry information about UAV flight session where
                each line correlates to each frame from cube

            Returns
            --------
            bands: np.ndarray
                Building result - hypercube from UAV footage
    """
    x, y, z = cube.shape  # in gaidel_legacy it's x=m, y=k, z=n
    gps = pd.read_csv(gps_filename, delimiter=CSV_DELIMITER)
    gps = gps.loc[gps[HEADER_CAM_ID] == GPS_HYPERCAM_FRAME].head(x)

    bands = interpolate(cube,
                        latitude=gps[HEADER_X].tolist(),
                        longitude=gps[HEADER_Y].tolist(),
                        rel_alt=gps[HEADER_REL_ALT].tolist(),
                        angle=gps[HEADER_ANGLE].tolist())

    bands = list(map(blur_band, [bands[:, :, i] for i in range(z)]))
    bands = np.array(bands)
    bands = np.transpose(bands, (2, 1, 0))
    return bands
# -------------------------------------------------------------------------------------------------------------------------------


# TODO interpolate doesn't use gps data, but inner computation. Define this more clearly
def interpolate(cube: np.ndarray,
                latitude: List,
                longitude: List,
                rel_alt: List,
                angle: List) -> np.ndarray:
    """
        interpolate(cube, latitude, longitude, rel_alt, angle)

            This function initiate calculating lines coordinate for hypercube
            based on telemetry data (latitude, longitude, rel_alt, angle)
            After this initiate KNN training for interpolation, finds nearest distance
            in prediction and checks if DISTANCE_LIMIT greater than the nearest neighbor
            and resize prediction relative with TARGET_RESOLUTION const

        Parameters:
        -----------
            cube: np.ndarray
                Preprocessing array with shape (nums_frames, width, height)

            latitude: list
                list of floats from metadata file described UAV latitude
            
            longitude: list
                list of floats from metadata file described UAV longitude

            rel_alt: list
                list of floats from metadata file described UAV altitude

            angle: list
                list of floats from metadata file described UAV angle
        
        Returns:
            3D-Array hypercube 
        
    """
    n, _, k = cube.shape

    rel_alt = calculate_rel_alt(rel_alt)
    latitude, longitude = calculate_lat_lon(latitude, longitude, rel_alt, angle)

    x, y, z = coordinates_for_frame(cube, latitude, longitude, rel_alt, angle)

    model = knn_for_interpolate(x, y, z)
    test, n_target, m_target = generate_test_points(x, y)
    prediction = model.predict(test)
    nearest_dit, _ = model.kneighbors(test, n_neighbors=1, return_distance=True)

    # check if the nearest distance is greater than the limit
    prediction[nearest_dit[:, 0] > DISTANCE_LIMIT, :] = np.zeros(k)

    prediction = prediction.reshape(m_target, n_target, k)

    return prediction
# ----------------------------------------------------------------------------------------------------------------------


def calculate_rel_alt(rel_alt: List) -> List:
    """
        calculate_rel_alt(rel_alt)

            For every frame calculating the altitude in relation to the camera pitch

        Parameters:
        ------------
        rel_alt: list
            list of floats from metadata file described UAV altitude

        Returns:
        -----------
            updated list rel_alts
    """

    return [alt / COS_PITCH for alt in rel_alt]
# -------------------------------------------------------------------------------------------------------------------------------


def calculate_lat_lon(latitude: List[float],
                      longitude: List[float],
                      rel_alt: List[float],
                      angle: List[float]):
    """
        calculate_lat_lon(latitude, longitude, rel_alt, angle)

            It calculates the position in 2D plane of the current frame
            in relation camera pitch, rel_alt and angle.

        Parameters:
        ------------------
        latitude: list
            list of floats from metadata file described UAV latitude
        
        longitude: list
            list of floats from metadata file described UAV longitude

        rel_alt: list
            list of floats from metadata file described UAV altitude

        angle: list
            list of floats from metadata file described UAV angle

        Returns:
            updated list of latitude 
            updated list of longitude

    """
    latitude = [lat + alt * TAN_PITCH * math.cos(ang) for lat, alt, ang in zip(latitude, rel_alt, angle)]
    longitude = [lon + alt * TAN_PITCH * math.sin(ang) for lon, alt, ang in zip(longitude, rel_alt, angle)]

    return latitude, longitude
# ----------------------------------------------------------------------------------------------------------------------


def coordinates_for_frame(cube: np.ndarray,
                          latitude: List[float],
                          longitude: List[float],
                          rel_alt: List[float],
                          angle: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        coordinates_for_frame(cube, latitude, longitude, rel_alt, angle)
            It calculates the coordinates in the frame width amount
            for each frame, taking telemetry into self.
            
        Parameters:
        -----------
            cube: np.ndarray
                Preprocessing array with shape (nums_frames, width, height)
            
            latitude: list
                list of floats from metadata file described UAV latitude
            
            longitude: list
                list of floats from metadata file described UAV longitude
            
            rel_alt: list
                list of floats from metadata file described UAV altitude
            
            angle: list
                list of floats from metadata file described UAV angle

        Returns:
        ---------
            list x with len (width of frame * nums of frame)
            list y with len (width of frame * nums of frame)
            list z with shape(width of frame * nums of frame, height)
    """
    n, m, k = cube.shape

    distance = np.array(rel_alt) * CAMERA_TANGENT / 2.0

    np_lat = np.array(latitude)
    np_lon = np.array(longitude)
    np_ang = np.array(angle)

    # left bound angles
    angs_1 = np_ang + PI_2

    # right bound angles
    angs_2 = np_ang - PI_2

    x_1 = np_lat + distance * np.cos(angs_1)
    x_2 = np_lat + distance * np.cos(angs_2)

    y_1 = np_lon + distance * np.sin(angs_1)
    y_2 = np_lon + distance * np.sin(angs_2)

    x = np.linspace(x_1, x_2, m).T.reshape((n * m))
    y = np.linspace(y_1, y_2, m).T.reshape((n * m))

    z = cube.reshape((n * m, k))

    return x, y, z
# ----------------------------------------------------------------------------------------------------------------------


def knn_for_interpolate(x: np.ndarray,
                        y: np.ndarray,
                        z: np.ndarray) -> neighbors.KNeighborsRegressor:
    """
        knn_for_interpolate(x, y, z)

            It trains KNN Model on coordinate data for interpolation of hypercube

        Parameters:
        -----------
            x, y: list of coordinates for data

            z: 2D list of coordinates for label
        Returns:
        ----------
            The trained model
    """
    model = neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
    data = np.stack((np.array(x), np.array(y))).T
    model.fit(data, z)
    return model
# ----------------------------------------------------------------------------------------------------------------------


# TODO: Maybe we should separate calculating test points and search m/n_targets?
def generate_test_points(x: List, y: List) -> Tuple:
    """
        generate_test_points(x, y)

            Here we made test data for KNN Model and also
            calculate size of hypercube in relation given
            TARGET_RESOLUTION

        Parameters:
        -----------
            x and y lists of coordinates
        
        Returns:
        -----------
            test_points: list
                list with shape(len(coord),2) for prediction
            
            m_target and n_target: integers
                for final resolution of hypercube
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    n_target = TARGET_RESOLUTION
    m_target = int(n_target * (y_max - y_min) / (x_max - x_min))
    test_points = list(itertools.product(np.linspace(x_min, x_max, n_target), np.linspace(y_min, y_max, m_target)))
    return test_points, m_target, n_target
# ----------------------------------------------------------------------------------------------------------------------


def blur_image(img: np.ndarray) -> np.ndarray:
    return cv2.blur(img, BLUR_SHAPE)
# ----------------------------------------------------------------------------------------------------------------------
