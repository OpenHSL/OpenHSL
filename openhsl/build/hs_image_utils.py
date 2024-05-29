import copy
import cv2 as cv
import discorpy.post.postprocessing as post
import discorpy.proc.processing as proc
import enum
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Optional, Tuple


class BaseIntEnum(enum.IntEnum):
    """
    Base class for int enumeration
    """

    def describe(self):
        return self.name, self.value

    @classmethod
    def enum_names(cls):
        return [v.name for v in list(cls)]

    @classmethod
    def to_dict(cls):
        return {i.name: i.value for i in cls}


def compute_slit_angle(frame: np.ndarray, x: int, y: int, w: int, h: int,
                       threshold_value: int, threshold_type=0) -> Tuple[float, float, float]:
    slope = 1
    angle = 0.0
    intercept = 0
    frame_channel: Optional[np.ndarray] = None

    if len(frame.shape) == 2:
        frame_channel = copy.deepcopy(frame)
    elif len(frame.shape) == 3:
        frame_channel = frame[:, :, 1]

    frame_channel = frame_channel[y: y + h, x: x + w]

    frame_thresholded: Optional[np.ndarray] = None

    frame_thresholded = threshold_image(frame_channel, threshold_value, 255, threshold_type)
    points = np.argwhere(frame_thresholded > 0)
    ya = points[:, 0]
    xa = points[:, 1]
    regr = LinearRegression().fit(xa.reshape(-1, 1), ya)
    slope = float(regr.coef_)
    angle = float(np.arctan(slope) * 180 / np.pi)
    intercept = float(regr.intercept_ + y)

    return slope, angle, intercept


def compute_undistortion_coeffs(coeffs: np.ndarray, powers: np.ndarray, factors: np.ndarray, image_shape: Tuple[int],
                                center_xy: Optional[np.ndarray] = None, line_step: int = 40) -> np.ndarray:
    height, width = image_shape[0:2]
    xcenter = width // 2
    ycenter = height // 2

    if center_xy is not None:
        if len(center_xy) == 2:
            xcenter, ycenter = center_xy

    # Prepare powers and coeffs arrays
    j = 0
    powers_prep = []
    coeffs_prep = []
    factors_prep = np.array([10 ** f for f in factors])
    for i in range(powers[-1] + 1):
        if i in powers:
            powers_prep.append(factors_prep[j])
            coeffs_prep.append(coeffs[j])
            j += 1
        else:
            powers_prep.append(0)
            coeffs_prep.append(0)
    powers_prep = np.array(powers_prep)
    coeffs_prep = np.array(coeffs_prep)

    list_ffact = powers_prep * coeffs_prep

    # Backward
    hlines = np.int16(np.linspace(0, height, line_step))
    vlines = np.int16(np.linspace(0, width, line_step))
    ref_points = [[i - ycenter, j - xcenter] for i in hlines for j in vlines]
    list_bfact = proc.transform_coef_backward_and_forward(list_ffact, ref_points=ref_points)

    return np.array(list_bfact)


def contrast_image(image: np.ndarray, contrast_value: int, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if len(image.shape) == 3:
        image_equalized = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        image_equalized = copy.deepcopy(image)
    clahe = cv.createCLAHE(clipLimit=contrast_value, tileGridSize=tile_grid_size)
    image_equalized = clahe.apply(image_equalized)
    if len(image.shape) == 3:
        image_equalized = cv.cvtColor(image_equalized, cv.COLOR_GRAY2RGB)
    return image_equalized


def norn_min_max(array: np.ndarray):
    v_min = np.min(array)
    v_max = np.max(array)
    array_mm = copy.deepcopy(array)
    array_mm = (array_mm - v_min) / (v_max - v_min)
    return array_mm


def normalize_illumination(frame: np.ndarray, illumination_mask: np.ndarray) -> np.ndarray:
    frame_ilm_norm = copy.deepcopy(frame)
    max_v = float(np.iinfo(frame.dtype).max)
    frame_ilm_norm = frame_ilm_norm.astype(float) / max_v
    idx = illumination_mask > 0
    frame_ilm_norm[idx] = frame_ilm_norm[idx] / illumination_mask[idx]
    frame_ilm_norm[frame_ilm_norm > 1] = 1
    frame_ilm_norm = (frame_ilm_norm * max_v).astype(frame.dtype)
    return frame_ilm_norm


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[0:2]
    center_x, center_y = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    image_rotated = cv.warpAffine(image, rotation_matrix, (w, h))
    return image_rotated


def threshold_image(image: np.ndarray, threshold_value: int, max_value, threshold_type=0):
    _, image_thresholded = cv.threshold(image, threshold_value, max_value, threshold_type)
    return image_thresholded


def undistort_image(image: np.ndarray, undistortion_coeffs: List[float],
                    center_xy: Optional[List[int]] = None) -> np.ndarray:
    image_undistorted = copy.deepcopy(image)
    height, width = image_undistorted.shape[0:2]
    xcenter = width // 2
    ycenter = height // 2

    if center_xy is not None:
        if len(center_xy) == 2:
            xcenter, ycenter = center_xy

    if len(image_undistorted.shape) == 3:
        for i in range(3):
            image_undistorted[:, :, i] = post.unwarp_image_backward(image_undistorted[:, :, i],
                                                                    xcenter, ycenter, undistortion_coeffs)
    else:
        image_undistorted = post.unwarp_image_backward(image_undistorted, xcenter, ycenter, undistortion_coeffs)

    return image_undistorted
