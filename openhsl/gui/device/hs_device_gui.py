import csv
import itertools
import json
import sys
from PyQt6.QtCore import Qt, QDir, QFileInfo, QEvent, QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, \
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar, QPushButton, \
    QSlider, QSpinBox, QToolBar, QToolButton, QWidget
from PyQt6 import uic
from typing import Any, Dict, List
from openhsl.hs_device import HSDevice


class HSDeviceQ(HSDevice, QObject):
    def __init__(self):
        super(HSDevice, self).__init__()
        super(QObject, self).__init__()


# noinspection PyTypeChecker
class HSDeviceGUI(QMainWindow):
    def __init__(self):
        super(HSDeviceGUI, self).__init__()
        uic.loadUi('hs_device_mainwindow.ui', self)
        with open("./Resources/Dark.qss", 'r') as f:
            strings = f.read()
            self.setStyleSheet(strings)

        self.hsd = HSDeviceQ()
        self.t_hsd = QThread()
        self.t_hsd.start()
        self.hsd.moveToThread(self.t_hsd)

        # UI controls
        # Menu bar
        self.ui_menu_bar: QMenuBar = self.findChild(QMenuBar, "menubar")
        self.ui_file_menu: QMenu = self.findChild(QMenu, "file_menu")
        self.ui_file_open_action: QAction = self.findChild(QAction, "fileOpen_action")
        self.ui_file_exit_action: QAction = self.findChild(QAction, "fileExit_action")
        self.ui_help_menu: QMenu = self.findChild(QMenu, "help_menu")
        self.ui_help_about_action: QAction = self.findChild(QAction, "helpAbout_action")

        # Slit angle tab
        self.ui_slit_image_path_open_button: QPushButton = self.findChild(QPushButton, 'slitImagePathOpen_pushButton')
        # self.ui_slit_image_path_open_button.setIcon(QIcon(QPixmap("icons:three-dots.svg")))

    def closeEvent(self, event):
        self.t_hsd.exit()
        event.accept()


def main():
    import sass

    qss_str = sass.compile(filename="Resources/Dark.scss", output_style='expanded')
    with open("Resources/Dark.qss", 'w') as f:
        f.write(qss_str)

    QDir.addSearchPath('icons', './Resources/Images/')
    QDir.addSearchPath('resources', './Resources/')
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
    hsd_gui = HSDeviceGUI()
    hsd_gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()