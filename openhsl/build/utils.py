import datetime
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional


# TODO rename to path_exists
def dir_exists(path: str) -> bool:
    return Path(path).exists()


def get_current_date(date_format_str="%d.%m.%Y") -> str:
    return datetime.datetime.now().strftime(date_format_str)


def get_current_time(time_format_str="%H:%M:%S") -> str:
    return datetime.datetime.now().strftime(time_format_str)


def key_exists_in_dict(dict_var: dict, key: str) -> bool:
    return key in dict_var


def get_file_complete_name(path: str) -> str:
    p = Path(path)
    return p.name


def load_data(path: str, exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]


def load_dict_from_json(path: str) -> Optional[dict]:
    data: Optional[dict] = None
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data


def save_dict_to_json(data: dict, path: str) -> None:
    with open(path, 'w') as handle:
        json.dump(data, handle, indent=4)


# TODO move into hs_image_utils
def gaussian(length: int, mean: float, std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
