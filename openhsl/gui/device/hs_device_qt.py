from PyQt6.QtCore import QObject
from openhsl.hs_device import HSDevice

class HSDeviceQt(HSDevice, QObject):
    def __init__(self):
        HSDevice.__init__(self)
        QObject.__init__(self)
