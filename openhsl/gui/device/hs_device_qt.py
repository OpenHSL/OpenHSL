import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage
from typing import Optional
from openhsl.hs_device import HSDevice

class HSDeviceQt(QObject, HSDevice):
    send_slit_image = pyqtSignal(QImage)
    # send_slit_angle = pyqtSignal(float)
    # send_slit_offset = pyqtSignal(float)
    def __init__(self):
        super(HSDeviceQt, self).__init__()

        self.slit_image: Optional[np.ndarray] = None
        self.slit_image_to_send: Optional[np.ndarray] = None
        self.row_count = 0
        self.col_count = 0

    def read_slit_image(self, path: str):
        self.slit_image = cv.imread(path, cv.IMREAD_COLOR)
        self.col_count, self.row_count = self.slit_image.shape[0:2]
        self.slit_image_to_send = cv.cvtColor(self.slit_image, cv.COLOR_BGR2RGB)
        image_to_draw_qt = QImage(self.slit_image_to_send, self.slit_image_to_send.shape[1],
                                  self.slit_image_to_send.shape[0], QImage.Format.Format_RGB888)
        self.send_slit_image.emit(image_to_draw_qt)
