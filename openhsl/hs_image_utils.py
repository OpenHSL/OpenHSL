import copy
import cv2 as cv
import enum
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple


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


def threshold_image(image: np.ndarray, threshold_value: int, max_value, threshold_type = 0):
    _, image_thresholded = cv.threshold(image, threshold_value, max_value, threshold_type)
    return image_thresholded
