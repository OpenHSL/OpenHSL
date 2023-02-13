from pathlib import Path
import math
import numpy as np

def dir_exists(path: str) -> bool:
    return Path(path).exists()

def load_data(path: str, exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]

def gaussian(length: int, mean: float, std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std