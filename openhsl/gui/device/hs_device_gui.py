import csv
import ctypes
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
from openhsl.hs_device import HSDevice, HSDeviceType, HSROI, HSCalibrationWavelengthData
import openhsl.utils as utils


class HSDeviceQ(HSDevice, QObject):
    def __init__(self):
        HSDevice.__init__(self)
        QObject.__init__(self)


# noinspection PyTypeChecker
class HSDeviceGUI(QMainWindow):
    def __init__(self):
        super(HSDeviceGUI, self).__init__()
        uic.loadUi('hs_device_mainwindow.ui', self)
        self.setWindowIcon(QIcon("icons:openhsl.svg"))

        # Workaround for taskbar icon in Windows
        # See: https://stackoverflow.com/a/1552105
        openhsl_id = 'locus.openhsl.hs_device_gui.0.0.1'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(openhsl_id)

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
        self.ui_recent_devices_menu: QMenu = self.findChild(QMenu, "recentDevices_menu")
        self.ui_file_exit_action: QAction = self.findChild(QAction, "fileExit_action")
        self.ui_help_menu: QMenu = self.findChild(QMenu, "help_menu")
        self.ui_help_about_action: QAction = self.findChild(QAction, "helpAbout_action")

        # Slit angle tab
        self.ui_slit_image_path_open_button: QPushButton = self.findChild(QPushButton, 'slitImagePathOpen_pushButton')
        # self.ui_slit_image_path_open_button.setIcon(QIcon(QPixmap("icons:three-dots.svg")))
        # Settings tab
        self.ui_device_type_combobox: QComboBox = self.findChild(QComboBox, "deviceType_comboBox")
        self.ui_device_type_combobox.addItem("test1")
        self.ui_device_type_combobox.addItem("test2")
        self.ui_device_settings_path_line_edit: QLineEdit = self.findChild(QLineEdit, "deviceSettingsPath_lineEdit")
        self.ui_device_settings_path_save_button: QPushButton = self.findChild(QPushButton,
                                                                               'deviceSettingsPathOpen_pushButton')
        self.ui_device_settings_save_button: QPushButton = self.findChild(QPushButton,
                                                                          'deviceSettingsSave_pushButton')

        # Signal and slot connections
        # Slit angle tab
        self.ui_slit_image_path_open_button.clicked.connect(self.on_slit_image_path_button_clicked)
        # Settings tab
        self.ui_device_settings_path_save_button.clicked.connect(self.on_ui_device_settings_path_save_button_clicked)
        self.ui_device_settings_save_button.clicked.connect(self.on_ui_device_settings_save_button_clicked)

        self.slit_image_path = ""
        self.recent_device_settings_path_list = []
        self.ui_recent_device_settings_action_list: List[QAction] = []
        self.device_settings_path = ""
        self.device_settings_name = ""
        self.last_device_settings_path = ""
        self.settings_name = 'hs_device_gui_settings.json'
        self.settings_dict = self.initialize_settings_dict()

        self.prepare_ui()

    def prepare_ui(self):
        self.fill_device_type_combobox()
        # TODO maybe add default zeros
        self.hsd.roi = HSROI()
        # TODO remove
        wl_1 = HSCalibrationWavelengthData()
        wl_1.wavelength = 415
        wl_2 = HSCalibrationWavelengthData()
        wl_2.wavelength = 705
        self.hsd.calib_wavelength_data = [wl_1, wl_2]

    def fill_device_type_combobox(self):
        d = HSDeviceType.to_dict()
        for k, v in d.items():
            self.ui_device_type_combobox.addItem(k, v)

    def fill_recent_devices_menu(self):
        for path in self.recent_device_settings_path_list:
            action = QAction(path, self.ui_recent_devices_menu)
            self.ui_recent_devices_menu.addAction(action)
            self.ui_recent_device_settings_action_list.append(action)

    def initialize_settings_dict(self):
        settings_dict = {
            "program": "HSDeviceGUI",
            "generation_date": utils.get_current_date(),
            "recent_device_settings_path_list": self.recent_device_settings_path_list,
            "last_device_settings_path": self.device_settings_path,
        }
        return settings_dict

    def save_settings(self):
        self.recent_device_settings_path_list = list(set(self.recent_device_settings_path_list))
        self.settings_dict["generation_date"] = utils.get_current_date()
        self.settings_dict["generation_time"] = utils.get_current_time()
        self.settings_dict["recent_device_settings_path_list"] = self.recent_device_settings_path_list
        self.settings_dict["last_device_settings_path"] = self.last_device_settings_path

        with open(self.settings_name, 'w') as settings_file:
            json.dump(self.settings_dict, settings_file, indent=4)

    def load_settings(self):
        settings_filename = self.settings_name
        if utils.dir_exists(settings_filename):
            with open(settings_filename) as settings_file:
                self.settings_dict = json.load(settings_file)
            # Settings tab
            if utils.key_exists_in_dict(self.settings_dict, "recent_device_settings_path_list"):
                self.recent_device_settings_path_list = self.settings_dict["recent_device_settings_path_list"]
                self.fill_recent_devices_menu()

    def on_main_window_is_shown(self):
        self.load_settings()

    def on_slit_image_path_button_clicked(self):
        self.slit_image_path = QFileDialog.getOpenFileName(self, "Choose file", "", "Image file (*.bmp *.png *.jpg *.tif)")

    def on_ui_device_settings_path_save_button_clicked(self):
        self.device_settings_path, _filter = QFileDialog.getSaveFileName(self, "Save file", "",
                                                                         "Settings file (*.json)")

        if self.device_settings_path != "":
            self.ui_device_settings_path_line_edit.setText(self.device_settings_path)
            self.device_settings_name = utils.get_file_complete_name(self.device_settings_path)

    def on_ui_device_settings_save_button_clicked(self):
        utils.save_dict_to_json(self.hsd.to_dict(), self.device_settings_path)
        self.last_device_settings_path = self.device_settings_path
        # TODO rewrite
        self.recent_device_settings_path_list.append(self.last_device_settings_path)

    def closeEvent(self, event):
        self.t_hsd.exit()
        self.save_settings()
        event.accept()

    def showEvent(self, event):
        event.accept()
        # zero interval timer fires only when after all events in the queue are processed
        QTimer.singleShot(0, self.on_main_window_is_shown)


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