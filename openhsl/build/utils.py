import datetime
import json
import math
import numpy as np
import re
from pathlib import Path
from typing import List, Optional, Set, Union


def absolute_file_path(path: Union[str, Path]) -> str:
    p = Path(path)
    return str(p.absolute())


def current_date(date_format_str="%d.%m.%Y") -> str:
    return datetime.datetime.now().strftime(date_format_str)


def current_time(time_format_str="%H:%M:%S") -> str:
    return datetime.datetime.now().strftime(time_format_str)


def file_complete_name(path: Union[str, Path]) -> str:
    p = Path(path)
    return p.name


def file_path_list(input_dir: str, extension_list: Union[List[str], Set[str]], name_only=False) -> List[str]:
    file_path_list = []
    input_path = Path(input_dir)
    if input_path.is_dir():
        if input_path.exists():
            path_list = (p.resolve() for p in input_path.glob("**/*") if p.suffix[1:] in extension_list)

            for path in path_list:
                # FIXME - patch for Javier
                if name_only:
                    file_path_list.append(file_complete_name(path))
                else:
                    file_path_list.append(absolute_file_path(path))
            file_path_list = sorted(file_path_list, key=natural_key)

    return file_path_list


def key_exists_in_dict(dict_var: dict, key: str) -> bool:
    return key in dict_var


def load_data(path: str, exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]


def load_dict_from_json(path: str) -> Optional[dict]:
    data: Optional[dict] = None
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data


def natural_key(str_line):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', str_line)]


def path_exists(path: str) -> bool:
    return Path(path).exists()


def save_dict_to_json(data: dict, path: str) -> None:
    with open(path, 'w') as handle:
        json.dump(data, handle, indent=4)


def gaussian(length: int, mean: float, std: float) -> np.ndarray:
    """
    gaussian(length, mean, std)

        Returns gaussian 1D-kernel

        Parameters
        ----------
        length: int
            gaussian 1D-Kernel length
        mean: float
            "height" of gaussian
        std:
            "slope" of gaussian
        Returns
        -------
            np.ndarray

    """
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
