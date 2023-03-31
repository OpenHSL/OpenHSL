import enum
import numpy as np


class BaseIntEnum(enum.IntEnum):
    """
    Base class for int enumeration
    """
    def describe(self):
        return self.name, self.value

    @classmethod
    def enum_names(cls):
        return [v.name for v in list(cls)]


def get_slit_angle(frame: np.ndarray) -> float:
    pass
