import os
from glob import glob

def load_data(path: str, format: str):
    return sorted(glob(os.path.join(path, f"*{format}")))