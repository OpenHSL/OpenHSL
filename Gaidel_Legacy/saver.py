# -*- coding: utf-8 -*-
import datetime
import hashlib
import json
import os
import shutil
import cv2
import numpy

import Gaidel_Legacy.helper as helper
import Gaidel_Legacy.messages as messages
import Gaidel_Legacy.settings as settings

from sklearn import preprocessing


ARCHIVE = "zip"
FORMAT_NUMPY = "numpy"
FORMAT_PNG = "png"
HASH_CHUNK_SIZE = 4096
JSON_BANDS_NUMBER = "bands_number"
JSON_ORIGINAL_FILENAME = "filename_original"
JSON_SPECTRAL_FLIP = "spectral_flip"
LAYER_NAME = "name"
LAYER_WAVELENGTH = "wavelength"
MODE_READ = "rb"
MODE_WRITE = "w"
WAVELENGTH_RED = 633.0
WAVELENGTH_GREEN = 525.0
WAVELENGTH_BLUE = 489.0


class HyperMetadata(object):

    def __init__(self, filename, file_format, file_hash, rgb, size, layers, description="", name="", parameters=None):
        if parameters is None:
            parameters = {}
        now = datetime.datetime.now()
        self.id = int(now.microsecond)
        self.name = name
        self.description = description
        self.hash = file_hash
        self.filename = filename
        self.format = file_format
        self.parameters = parameters
        self.r = int(rgb[0])
        self.g = int(rgb[1])
        self.b = int(rgb[2])
        self.height = int(size[0])
        self.width = int(size[1])
        self.layers = [{LAYER_NAME: str(int(round(layer))), LAYER_WAVELENGTH: layer} for layer in layers]

    def save_json(self, filename):
        with open(filename, MODE_WRITE) as out:
            json.dump(self.__dict__, out)


def calc_hash(filename):
    md5 = hashlib.md5()
    with open(filename, MODE_READ) as fin:
        for chunk in iter(lambda: fin.read(HASH_CHUNK_SIZE), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_rgb_channels(layers):
    return [
        numpy.argmin((numpy.array(layers) - wavelength) ** 2) + 1 for wavelength in (
            WAVELENGTH_RED, WAVELENGTH_GREEN, WAVELENGTH_BLUE
        )
    ]


def make_json_metadata(spectrum, file_format, hyper_filename, json_filename, layers, description="", parameters=None):
    if parameters is None:
        parameters = {}
    if layers is None:
        layers = numpy.linspace(settings.WAVELENGTH_MIN, settings.WAVELENGTH_MAX, spectrum.shape[0])
    now = datetime.datetime.now()
    HyperMetadata(
        name=now.strftime(settings.FILENAME_TIME_PATTERN),
        description=description,
        filename=os.path.basename(hyper_filename),
        file_format=file_format,
        file_hash=calc_hash(hyper_filename),
        rgb=get_rgb_channels(layers),
        size=(spectrum.shape[1], spectrum.shape[2]),
        layers=layers,
        parameters=parameters,
    ).save_json(json_filename)


def save_slices(spectrum, path, basename, file_format, layers=None, original_path=None):
    if settings.VERBOSE:
        print(messages.SLICES_SAVING)
    scaler = preprocessing.MinMaxScaler((0, 255))
    helper.clear_path(path)
    if settings.ROTATE_TIMES != 0:
        spectrum = numpy.rot90(spectrum, settings.ROTATE_TIMES, axes=(1, 2))
    hyper_filename = os.path.join(path, basename)
    n, m, k = spectrum.shape
    if settings.SPECTRAL_FLIP:
        spectrum = numpy.flip(spectrum, axis=0)
    if settings.config.postprocessing_scaling_gray_levels:
        for j in range(spectrum.shape[0]):
            spectrum[j, :, :] = numpy.reshape(
                scaler.fit_transform(spectrum[j, :, :].reshape((m * k, 1)).astype(numpy.float64)),
                (m, k)
            )
    if settings.RESCALE_AUTO:
        m, k = settings.SHAPE_OUTPUT
        spectrum_new = numpy.zeros((n, k, m))
        for j in range(spectrum.shape[0]):
            spectrum_new[j, :, :] = cv2.resize(spectrum[j, :, :], settings.SHAPE_OUTPUT)
        spectrum = spectrum_new
    if settings.BLUR_AUTO:
        for j in range(spectrum.shape[0]):
            spectrum[j, :, :] = cv2.blur(spectrum[j, :, :], settings.BLUR_SHAPE)
    if settings.VERTICAL_FLIP:
        spectrum = numpy.flip(spectrum, axis=1)
    if settings.HORIZONTAL_FLIP:
        spectrum = numpy.flip(spectrum, axis=2)
    if file_format == FORMAT_PNG:
        for j in range(spectrum.shape[0]):
            cv2.imwrite(os.path.join(path, settings.IMAGE_PATTERN.format(j)), spectrum[j, :, :])
        archive_path = os.path.join(settings.PROJECT_DIR, settings.DIR_TEMP)
        hyper_filename = shutil.make_archive(os.path.join(archive_path, basename), ARCHIVE, path)
        shutil.rmtree(path)
        os.makedirs(path)
        shutil.copy2(hyper_filename, path)
        hyper_filename = os.path.join(path, os.path.basename(hyper_filename))
    elif file_format == FORMAT_NUMPY:
        hyper_filename = os.path.join(path, helper.change_extension(basename, settings.EXTENSION_NUMPY))
        numpy.save(hyper_filename, spectrum.astype(numpy.uint8))
    # make_json_metadata(
    #     spectrum,
    #     file_format,
    #     os.path.abspath(hyper_filename),
    #     os.path.join(path, helper.change_extension(hyper_filename, settings.EXTENSION_JSON)),
    #     layers,
    #     basename,
    #     parameters={
    #         JSON_ORIGINAL_FILENAME: original_path,
    #         JSON_BANDS_NUMBER: settings.config.spectral_bands_number,
    #         JSON_SPECTRAL_FLIP: settings.SPECTRAL_FLIP,
    #     }
    # )
    return spectrum
