from pathlib import Path
import math
import numpy as np
from sklearn.decomposition import PCA

def dir_exists(path: str) -> bool:
    return Path(path).exists()
# ----------------------------------------------------------------------------------------------------------------------


def load_data(path: str,
              exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]
# ----------------------------------------------------------------------------------------------------------------------


def gaussian(length: int,
             mean: float,
             std: float) -> np.ndarray:
    return np.exp(-((np.arange(0, length) - mean) ** 2) / 2.0 / (std ** 2)) / math.sqrt(2.0 * math.pi) / std
# ----------------------------------------------------------------------------------------------------------------------


def applyPCA(X: np.ndarray,
             numComponents: int = 75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True, random_state=131)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca
# ----------------------------------------------------------------------------------------------------------------------


def padWithZeros(X: np.ndarray,
                 margin: int = 2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
# ----------------------------------------------------------------------------------------------------------------------
