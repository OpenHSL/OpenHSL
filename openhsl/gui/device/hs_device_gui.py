import csv
import itertools
import json
import sys
from PyQt6.QtCore import Qt, QDir, QFileInfo, QEvent, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, \
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel, QLineEdit, QMainWindow, QMenu, QPushButton, QSlider, \
    QSpinBox, QToolBar, QToolButton, QWidget
from PyQt6 import uic
from typing import Any, Dict, List


# noinspection PyTypeChecker
class HSDeviceGUI(QMainWindow):
    def __init__(self):
        super(HSDeviceGUI, self).__init__()
        uic.loadUi('hs_device_mainwindow.ui', self)


def main():
    # QDir.addSearchPath('icons', './Resources/Images/')
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
    hsd_gui = HSDeviceGUI()
    hsd_gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()