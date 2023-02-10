from pathlib import Path

def dir_exists(path: str) -> bool:
    return Path(path).exists()

def load_data(path: str, exts: list) -> list:
    return [str(p) for p in Path(path).glob("*") if p.suffix[1:] in exts]