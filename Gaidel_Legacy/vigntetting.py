import numpy
import json
import Gaidel_Legacy.settings as settings

ACTION_STORE_TRUE = "store_true"
PARAMETER_VIGNETTING = "vignetting"

class VignettingFixer(object):

    def __init__(self, config_filename):
        with open(config_filename, settings.MODE_READ) as fin:
            self.coefficients = numpy.array(json.load(fin)[PARAMETER_VIGNETTING], dtype=numpy.float64)

    def fix(self, frame):
        if frame.shape[1] == self.coefficients.shape[0]:
            return numpy.clip(frame * self.coefficients, 0, settings.COLOR_MAX)
        else:
            return frame