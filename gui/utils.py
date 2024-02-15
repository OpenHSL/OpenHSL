import os
import cv2
import h5py
import json
import torch
import numpy as np
from time import localtime
import scipy.io as sio


def get_file_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_file_extension(filepath):
    return os.path.splitext(os.path.basename(filepath))[1]


def get_file_directory(filepath):
    return os.path.dirname(filepath)


def create_dir_if_not_exist(path_to_dir):
    path_to_dir = path_to_dir.replace("\\", "/")
    path_to_dir = path_to_dir.replace("//", "/")
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)


def get_date_time():
    local_time = localtime()
    year = local_time[0]
    month = local_time[1]
    day = local_time[2]
    hour = local_time[3]
    minute = local_time[4]
    second = local_time[5]

    date = f"{year}_{month}_{day}"
    time = f"{hour}_{minute}_{second}"
    return date, time


def save_json_dict(data, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write(json.dumps(data))


def get_high_contrast(image: np.array):
    image = image.astype(np.float32)
    image = image - np.min(image)
    image = image / np.max(image)
    image = image * 255
    return image.astype(np.uint8)


def scale_image(image, scale_factor):
    scaled_height = int(image.shape[0] * scale_factor / 100)
    scaled_width = int(image.shape[1] * scale_factor / 100)
    return cv2.resize(image, (scaled_width, scaled_height))


def request_keys_from_mat_file(pathfile):
    mat = sio.loadmat(pathfile)
    keys = []
    for key in mat.keys():
        if not key.startswith("__"):
            keys.append(key)
    return keys


def request_keys_from_h5_file(pathfile):
    keys = []
    with h5py.File(pathfile, 'r') as f:
        f.visit(keys.append)
    return keys


def get_gpu_info():
    devices = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(torch.cuda.get_device_name(i))
    devices.append("cpu")
    return devices