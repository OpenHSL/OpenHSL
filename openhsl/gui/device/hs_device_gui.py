import copy
import ctypes
import json
import matplotlib as mpl
import sys

import numpy as np
from PyQt6.QtCore import Qt, QDir, QEvent, QLineF, QObject, QPointF, QRect, QRectF, QSignalMapper, QSize, QThread, \
    QTimer, pyqtSignal, pyqtSlot, QModelIndex
from PyQt6.QtGui import QAction, QActionGroup, QBrush, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, \
    QPolygonF, QStandardItemModel
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, \
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem, \
    QGraphicsTextItem, QGraphicsScene, QGraphicsView, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar, \
    QMessageBox, QPushButton, QSlider, QSpinBox, QTableView, QTableWidget, QTableWidgetItem, QToolBar, QToolButton, \
    QWidget
from PyQt6 import uic
from typing import Any, Dict, List, Optional
from openhsl.build.hs_device import HSDevice, HSDeviceType, HSCalibrationSlitData, HSCalibrationWavelengthData
from openhsl.gui.device.custom_controls import EquationParamsTableHeaderViewHorizontal, \
    EquationParamsTableHeaderViewVertical, EquationParamsTableModel, EquationParamsTableItem, HSGraphicsView, \
    WavelengthCalibrationTableItem, WavelengthCalibrationTableModel
from openhsl.gui.device.hs_device_qt import HSDeviceQt
import openhsl.gui.device.utils as hsd_gui_utils
import openhsl.build.utils as utils


# noinspection PyTypeChecker
class HSDeviceGUI(QMainWindow):
    read_slit_image = pyqtSignal(str)
    threshold_slit_image = pyqtSignal()
    compute_slit_angle = pyqtSignal(QRectF)
    rotate_bd_slit_image = pyqtSignal()
    contrast_bd_slit_image = pyqtSignal()
    draw_distortion_grid = pyqtSignal(bool)
    undistort_slit_image = pyqtSignal(bool)
    compute_bd_slit_center = pyqtSignal(QRectF, int)
    read_wl_image_dir = pyqtSignal(str)
    read_wl_image = pyqtSignal(int, bool, bool, bool)

    def __init__(self):
        super(HSDeviceGUI, self).__init__()
        uic.loadUi('hs_device_mainwindow.ui', self)
        self.setWindowIcon(QIcon("icons:OpenHSL_logo.png"))

        # Workaround for taskbar icon in Windows
        # See: https://stackoverflow.com/a/1552105
        openhsl_id = 'locus.openhsl.hs_device_gui.1.0.0'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(openhsl_id)

        self.bdew = QWidget()
        uic.loadUi('bd_equation_form.ui', self.bdew)
        self.bdew.setWindowIcon(QIcon("icons:OpenHSL_logo.png"))

        self.wcdw = QWidget()
        uic.loadUi('wcd_form.ui', self.wcdw)
        self.wcdw.setWindowIcon(QIcon("icons:OpenHSL_logo.png"))

        qss_path = "./Resources/Dark.qss"
        # qss_path = "./Resources/Grey.qss"
        self.stylesheet = ""

        with open(qss_path, 'r') as f:
            self.stylesheet = f.read()
            self.setStyleSheet(self.stylesheet)
            self.bdew.setStyleSheet(self.stylesheet)
            self.wcdw.setStyleSheet(self.stylesheet)

        self.hsd = HSDeviceQt()
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
        self.ui_slit_angle_graphics_view: HSGraphicsView = self.findChild(HSGraphicsView, 'slitAngle_graphicsView')
        self.ui_slit_image_threshold_value_checkbox: QCheckBox = \
            self.findChild(QCheckBox, 'slitImageThresholdValue_checkBox')
        self.ui_slit_image_threshold_value_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'slitImageThresholdValue_horizontalSlider')
        self.ui_slit_image_threshold_value_spinbox: QSpinBox = \
            self.findChild(QSpinBox, 'slitImageThresholdValue_spinBox')
        self.ui_slit_angle_horizontal_slider: QSlider = self.findChild(QSlider, 'slitAngle_horizontalSlider')
        self.ui_slit_angle_double_spinbox: QDoubleSpinBox = self.findChild(QDoubleSpinBox, 'slitAngle_doubleSpinBox')
        self.ui_slit_intercept_horizontal_slider: QSlider = self.findChild(QSlider, 'slitIntercept_horizontalSlider')
        self.ui_slit_intercept_double_spinbox: QDoubleSpinBox = \
            self.findChild(QDoubleSpinBox, 'slitIntercept_doubleSpinBox')
        self.ui_slit_image_path_open_button: QPushButton = self.findChild(QPushButton, 'slitImagePathOpen_pushButton')
        self.ui_slit_image_path_line_edit: QLineEdit = self.findChild(QLineEdit, 'slitImagePath_lineEdit')
        self.ui_load_slit_image_button: QPushButton = self.findChild(QPushButton, 'loadSlitImage_pushButton')
        self.ui_calc_slit_angle_button: QPushButton = self.findChild(QPushButton, 'calcSlitAngle_pushButton')
        # Barrel distortion tab - BDT
        self.ui_bdt_graphics_view: HSGraphicsView = self.findChild(HSGraphicsView, 'bdt_graphicsView')
        self.ui_bdt_apply_rotation_checkbox: QCheckBox = self.findChild(QCheckBox, 'bdtApplyRotation_checkBox')
        self.ui_bdt_slit_image_contrast_value_checkbox: QCheckBox = \
            self.findChild(QCheckBox, 'bdtSlitImageContrastValue_checkBox')
        self.ui_bdt_slit_image_contrast_value_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'bdtSlitImageContrastValue_horizontalSlider')
        self.ui_bdt_slit_image_contrast_value_spinbox: QSpinBox = \
            self.findChild(QSpinBox, 'bdtSlitImageContrastValue_spinBox')
        self.ui_bdt_distortion_grid_checkbox: QCheckBox = self.findChild(QCheckBox, 'bdtDistortionGrid_checkBox')
        self.ui_bdt_equation_view_label: QLabel = self.findChild(QLabel, 'bdtEquationView_label')
        self.ui_bdt_equation_set_button: QPushButton = self.findChild(QPushButton, 'bdtEquationSet_pushButton')
        self.ui_bdt_get_slit_center_button: QPushButton = self.findChild(QPushButton, 'bdtGetSlitCenter_pushButton')
        self.ui_bdt_undistort_image_button: QPushButton = self.findChild(QPushButton, 'bdtUndistortImage_pushButton')
        # BDT: Barrel distortion equation window
        self.ui_bdew_equation_table_view: QTableView = self.bdew.findChild(QTableView, 'equation_tableView')
        self.ui_bdew_equation_table_view_model: EquationParamsTableModel = None
        self.ui_bdew_center_x_spinbox: QSpinBox = self.bdew.findChild(QSpinBox, 'centerX_spinBox')
        self.ui_bdew_center_y_spinbox: QSpinBox = self.bdew.findChild(QSpinBox, 'centerY_spinBox')
        self.ui_bdew_grid_tile_size_spinbox: QSpinBox = self.bdew.findChild(QSpinBox, 'gridTileSize_spinBox')
        self.ui_bdew_polynomial_degree_spinbox: QSpinBox = self.bdew.findChild(QSpinBox, 'polynomialDegree_spinBox')
        self.ui_bdew_equation_header_view_vertical: Optional[EquationParamsTableHeaderViewVertical] = None
        self.ui_bdew_equation_header_view_horizontal: Optional[EquationParamsTableHeaderViewHorizontal] = None
        # Wavelengths tab
        self.ui_wt_graphics_view: HSGraphicsView = self.findChild(HSGraphicsView, 'wt_graphicsView')
        self.ui_wt_image_dir_path_open_button: QPushButton = \
            self.findChild(QPushButton, 'wtImageDirPathOpen_pushButton')
        self.ui_wt_image_dir_path_line_edit: QLineEdit = self.findChild(QLineEdit, 'wtImageDirPath_lineEdit')
        self.ui_wt_current_wavelength_image_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'wtCurrentWavelengthImage_horizontalSlider')
        self.ui_wt_current_wavelength_image_spinbox: QSpinBox = \
            self.findChild(QSpinBox, 'wtCurrentWavelengthImage_spinBox')
        self.ui_wt_apply_rotation_checkbox: QCheckBox = self.findChild(QCheckBox, 'wtApplyRotation_checkBox')
        self.ui_wt_apply_undistortion_checkbox: QCheckBox = self.findChild(QCheckBox, 'wtApplyUndistortion_checkBox')
        self.ui_wt_apply_contrast_preview_checkbox: QCheckBox = \
            self.findChild(QCheckBox, 'wtApplyContrastPreview_checkBox')
        self.ui_wt_contrast_preview_value_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'wtContrastPreviewValue_horizontalSlider')
        self.ui_wt_contrast_preview_value_spinbox: QSpinBox = self.findChild(QSpinBox, 'wtContrastPreviewValue_spinBox')
        self.ui_wt_wavelength_calibration_data_window_show_button: QPushButton = \
            self.findChild(QPushButton, 'wtWavelengthCalibrationDataWindowShow_pushButton')
        # WT: Wavelength calibration data window
        self.ui_wcdw_wavelength_line_y_coord_horizontal_slider: QSlider = \
            self.wcdw.findChild(QSlider, 'wavelengthLineYCoord_horizontalSlider')
        self.ui_wcdw_wavelength_line_y_coord_spinbox: QSpinBox = \
            self.wcdw.findChild(QSpinBox, 'wtWavelengthLineYCoord_spinBox')
        self.ui_wcdw_spectrum_origin_x_coord_horizontal_slider: QSlider = \
            self.wcdw.findChild(QSlider, 'wtSpectrumOriginXCoord_horizontalSlider')
        self.ui_wcdw_spectrum_origin_x_coord_spinbox: QSpinBox = \
            self.wcdw.findChild(QSpinBox, 'wtSpectrumOriginXCoord_spinBox')
        self.ui_wcdw_spectrum_origin_y_coord_horizontal_slider: QSlider = \
            self.wcdw.findChild(QSlider, 'wtSpectrumOriginYCoord_horizontalSlider')
        self.ui_wcdw_spectrum_origin_y_coord_spinbox: QSpinBox = \
            self.wcdw.findChild(QSpinBox, 'wtSpectrumOriginYCoord_spinBox')
        self.ui_wcdw_spectrum_width_horizontal_slider: QSlider = \
            self.wcdw.findChild(QSlider, 'wtSpectrumWidth_horizontalSlider')
        self.ui_wcdw_spectrum_width_spinbox: QSpinBox = self.wcdw.findChild(QSpinBox, 'wtSpectrumWidth_spinBox')
        self.ui_wcdw_spectrum_height_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'wtSpectrumHeight_horizontalSlider')
        self.ui_wcdw_spectrum_height_spinbox: QSpinBox = self.wcdw.findChild(QSpinBox, 'wtSpectrumHeight_spinBox')
        self.ui_wcdw_wavelength_table_view: QTableView = self.wcdw.findChild(QTableView, 'wtWavelength_tableView')
        self.ui_wcdw_wavelength_table_view_model: WavelengthCalibrationTableModel = None
        # Settings tab
        self.ui_device_type_combobox: QComboBox = self.findChild(QComboBox, "deviceType_comboBox")
        self.ui_device_settings_path_line_edit: QLineEdit = self.findChild(QLineEdit, "deviceSettingsPath_lineEdit")
        self.ui_device_settings_path_save_button: QPushButton = \
            self.findChild(QPushButton, 'deviceSettingsPathSave_pushButton')
        self.ui_device_settings_save_button: QPushButton = \
            self.findChild(QPushButton, 'deviceSettingsSave_pushButton')

        # Signal and slot connections
        # Main window
        self.ui_file_open_action.triggered.connect(self.on_ui_file_open_action_triggered)
        self.ui_file_exit_action.triggered.connect(self.on_ui_file_exit_action_triggered)
        # Slit angle tab
        self.ui_slit_image_threshold_value_checkbox.clicked.connect(
            self.on_ui_slit_image_threshold_value_checkbox_clicked)
        self.ui_slit_image_threshold_value_horizontal_slider.valueChanged.connect(
            self.on_ui_slit_image_threshold_value_horizontal_slider_value_changed)
        self.ui_slit_image_threshold_value_spinbox.valueChanged.connect(
            self.on_ui_slit_image_threshold_value_spinbox_value_changed)
        self.ui_slit_angle_horizontal_slider.valueChanged.connect(self.on_ui_slit_angle_horizontal_slider_value_changed)
        self.ui_slit_angle_double_spinbox.valueChanged.connect(self.on_ui_slit_angle_double_spinbox_value_changed)
        self.ui_slit_intercept_horizontal_slider.valueChanged.connect(
            self.on_ui_slit_intercept_horizontal_slider_value_changed)
        self.ui_slit_intercept_double_spinbox.valueChanged.connect(
            self.on_ui_slit_intercept_double_spinbox_value_changed)
        self.ui_slit_image_path_open_button.clicked.connect(self.on_ui_slit_image_path_button_clicked)
        self.ui_load_slit_image_button.clicked.connect(self.on_ui_load_slit_image_button_clicked)
        self.ui_calc_slit_angle_button.clicked.connect(self.on_ui_calc_slit_angle_button_clicked)
        self.read_slit_image.connect(self.hsd.on_read_slit_image, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_slit_image.connect(self.receive_slit_image, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_slit_preview_image.connect(self.receive_slit_preview_image, Qt.ConnectionType.QueuedConnection)
        self.threshold_slit_image.connect(self.hsd.on_threshold_slit_image, Qt.ConnectionType.QueuedConnection)
        self.compute_slit_angle.connect(self.hsd.on_compute_slit_angle)
        self.hsd.compute_slit_angle_finished.connect(self.on_compute_slit_angle_finished)
        self.hsd.adjust_slit_angle_range.connect(self.on_adjust_slit_angle_range)
        self.hsd.adjust_slit_intercept_range.connect(self.on_adjust_slit_intercept_range)
        # Barrel distortion tab
        self.ui_bdt_apply_rotation_checkbox.clicked.connect(self.on_ui_bdt_apply_rotation_checkbox_clicked)
        self.ui_bdt_slit_image_contrast_value_checkbox.clicked.connect(
            self.on_ui_bdt_slit_image_contrast_value_checkbox_clicked)
        self.ui_bdt_slit_image_contrast_value_horizontal_slider.valueChanged.connect(
            self.on_ui_bdt_slit_image_contrast_value_horizontal_slider_value_changed)
        self.ui_bdt_slit_image_contrast_value_spinbox.valueChanged.connect(
            self.on_ui_bdt_slit_image_contrast_value_spinbox_value_changed)
        self.ui_bdt_distortion_grid_checkbox.clicked.connect(self.on_ui_bdt_distortion_grid_checkbox_clicked)
        self.rotate_bd_slit_image.connect(self.hsd.on_rotate_bd_slit_image, Qt.ConnectionType.QueuedConnection)
        self.contrast_bd_slit_image.connect(self.hsd.on_contrast_bd_slit_image, Qt.ConnectionType.QueuedConnection)
        self.draw_distortion_grid.connect(self.hsd.on_draw_distortion_grid, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_image_rotated.connect(
            self.on_receive_bd_slit_image_rotated, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_image_contrasted.connect(self.on_receive_bd_slit_image_contrasted,
                                                       Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_distortion_grid_image.connect(self.on_receive_bd_distortion_grid_image,
                                                       Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_undistorted_slit_image.connect(self.on_receive_bd_undistorted_slit_image,
                                                        Qt.ConnectionType.QueuedConnection)
        self.ui_bdt_equation_set_button.clicked.connect(self.on_ui_bdt_equation_set_button_clicked)
        self.ui_bdt_get_slit_center_button.clicked.connect(self.on_ui_bdt_get_slit_center_button_clicked)
        self.compute_bd_slit_center.connect(self.hsd.on_compute_bd_slit_center, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_center.connect(self.on_receive_bd_slit_center, Qt.ConnectionType.QueuedConnection)
        self.ui_bdew_center_x_spinbox.editingFinished.connect(self.on_ui_bdew_center_x_spinbox_editing_finished)
        self.ui_bdew_center_y_spinbox.editingFinished.connect(self.on_ui_bdew_center_y_spinbox_editing_finished)
        self.ui_bdew_grid_tile_size_spinbox.editingFinished.connect(
            self.on_ui_bdew_grid_tile_size_spinbox_editing_finished)
        self.ui_bdew_polynomial_degree_spinbox.editingFinished.connect(
            self.on_ui_bdew_polynomial_degree_spinbox_editing_finished)
        self.ui_bdt_undistort_image_button.clicked.connect(self.on_ui_bdt_undistort_image_button_clicked)
        self.undistort_slit_image.connect(self.hsd.on_undistort_slit_image, Qt.ConnectionType.QueuedConnection)
        # Wavelengths tab
        self.ui_wt_image_dir_path_open_button.clicked.connect(self.on_ui_wt_image_dir_path_open_button_clicked)
        self.read_wl_image_dir.connect(self.hsd.on_read_wl_image_dir, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_wl_image_count.connect(self.on_receive_wl_image_count, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_wl_image.connect(self.on_receive_wl_image, Qt.ConnectionType.QueuedConnection)
        self.read_wl_image.connect(self.hsd.on_read_wl_image, Qt.ConnectionType.QueuedConnection)
        self.ui_wt_current_wavelength_image_horizontal_slider.valueChanged.connect(
            self.on_ui_wt_current_wavelength_image_horizontal_slider_value_changed)
        self.ui_wt_current_wavelength_image_spinbox.valueChanged.connect(
            self.on_ui_wt_current_wavelength_image_spinbox_value_changed)
        self.ui_wt_apply_rotation_checkbox.clicked.connect(self.on_ui_wt_apply_rotation_checkbox_clicked)
        self.ui_wt_apply_undistortion_checkbox.clicked.connect(self.on_ui_wt_apply_undistortion_checkbox_clicked)
        self.ui_wt_apply_contrast_preview_checkbox.clicked.connect(
            self.on_ui_wt_apply_contrast_preview_checkbox_clicked)
        self.ui_wt_contrast_preview_value_horizontal_slider.valueChanged.connect(
            self.on_ui_wt_contrast_preview_value_horizontal_slider_value_changed)
        self.ui_wt_contrast_preview_value_spinbox.valueChanged.connect(
            self.on_ui_wt_contrast_preview_value_spinbox_value_changed)
        self.ui_wt_wavelength_calibration_data_window_show_button.clicked.connect(
            self.on_ui_wt_wavelength_calibration_data_window_show_button_clicked)
        # Settings tab
        self.ui_device_settings_path_save_button.clicked.connect(self.on_ui_device_settings_path_save_button_clicked)
        self.ui_device_settings_save_button.clicked.connect(self.on_ui_device_settings_save_button_clicked)

        # Slit angle tab graphics
        self.slit_angle_graphics_scene = QGraphicsScene(self)
        self.slit_image_path = ""
        self.slit_image_qt: Optional[QImage] = None
        self.slit_graphics_pixmap_item = QGraphicsPixmapItem()
        self.slit_graphics_line_item = QGraphicsLineItem()
        self.slit_graphics_roi_rect_item = QGraphicsRectItem()
        self.slit_graphics_marquee_area_rect_item = QGraphicsRectItem()
        self.slit_graphics_text_item = QGraphicsTextItem()
        self.slit_angle_slider_mult = 10000000

        # Barrel distortion tab graphics
        self.bdt_graphics_scene = QGraphicsScene(self)
        self.bdt_graphics_pixmap_item = QGraphicsPixmapItem()
        self.bdt_graphics_marquee_area_rect_item = QGraphicsRectItem()
        self.bdt_graphics_center_x_line_item = QGraphicsLineItem()
        self.bdt_graphics_center_y_line_item = QGraphicsLineItem()
        self.bdt_slit_image_qt: Optional[QImage] = None
        self.bdt_slit_image_rotated_qt: Optional[QImage] = None
        self.bdt_slit_image_contrasted_qt: Optional[QImage] = None

        # Wavelengths tab graphics
        self.wt_graphics_scene = QGraphicsScene(self)
        self.wt_wavelength_image_dir_path = ""
        self.wt_wavelength_image_qt: Optional[QImage] = None
        self.wt_graphics_pixmap_item = QGraphicsPixmapItem()
        self.wt_graphics_slit_line_item = QGraphicsLineItem()
        self.wt_graphics_wavelength_line_item = QGraphicsLineItem()

        # TODO add mutex locker
        # TODO List[bool], for each tab different val?
        self.init_after_load_device_settings = False

        self.recent_device_settings_path_list = []
        self.ui_recent_device_settings_action_list: List[QAction] = []
        # Recent device settings action triggered signal mapper for actions
        self.recent_device_settings_action_triggered_signal_mapper = QSignalMapper(self)
        self.device_settings_path = ""
        self.device_settings_name = ""
        self.device_settings_dict = dict()
        self.last_device_settings_path = ""
        self.settings_name = 'hs_device_gui_settings.json'
        self.settings_dict = self.initialize_settings_dict()

        self.latex_pixmap_color = '#d0d0d0'
        self.latex_pixmap_selected_color = '#d0d0d0'
        self.latex_font_size = None

        self.prepare_ui()
        # self.installEventFilter(self)

    def render_latex_images(self):
        dir_path = utils.absolute_file_path(QDir.searchPaths('icons_gen')[0])
        mpl_ver = int(mpl.__version__.replace(".", ""))
        poly_deg = self.ui_bdew_polynomial_degree_spinbox.maximum()

        # Equation table labels
        image = QImage(f'icons_gen:bdet-header-vert-{poly_deg}.png')

        if image.rect().isEmpty():
            latex_labels = ["$1$"]
            latex_labels_selected = ["$\mathbf{1}$"]

            latex_bf = [r'$\bf{r}^{\bf{', r'}}$']

            if mpl_ver >= 380:
                latex_bf = [r'$\mathbfit{r}^{\bf{', r'}}$']

            for i in range(1, poly_deg + 1):
                latex_labels.append(f"$r^{{{i}}}$")
                latex_labels_selected.append(latex_bf[0] + f'{i}' + latex_bf[1])

            for i in range(len(latex_labels)):
                hsd_gui_utils.latex_to_file(f'{dir_path}/bdet-header-vert-{i}.png',
                                            latex_labels[i], self.latex_pixmap_color, self.latex_font_size)
                hsd_gui_utils.latex_to_file(f'{dir_path}/bdet-header-vert-selected-{i}.png',
                                            latex_labels_selected[i], self.latex_pixmap_color, self.latex_font_size)

        image = QImage('icons_gen:bdet-header-hor-1.png')

        if image.rect().isEmpty():
            latex_labels = [r'$\mathrm{Coefficient} \; k$', r'$\mathrm{Factor} \; 10^x$']
            latex_labels_selected = [r'$\mathbf{Coefficient} \; \mathbf{k}$',
                                     r'$\mathbf{Factor} \; \mathbf{10^x}$']

            if mpl_ver >= 380:
                latex_labels_selected = [r'$\mathbf{Coefficient} \; \mathbfit{k}$',
                                         r'$\mathbf{Factor} \; \mathbf{10^\mathbfit{x}}$']

            for i in range(len(latex_labels)):
                hsd_gui_utils.latex_to_file(f'{dir_path}/bdet-header-hor-{i}.png',
                                            latex_labels[i], self.latex_pixmap_color, self.latex_font_size)
                hsd_gui_utils.latex_to_file(f'{dir_path}/bdet-header-hor-selected-{i}.png',
                                            latex_labels_selected[i], self.latex_pixmap_color, self.latex_font_size)

    def prepare_ui(self):
        self.render_latex_images()
        self.fill_device_type_combobox()
        # TODO maybe add default zeros
        self.hsd.calib_slit_data = HSCalibrationSlitData()
        # TODO remove
        wl_1 = HSCalibrationWavelengthData()
        wl_1.wavelength = 415
        wl_2 = HSCalibrationWavelengthData()
        wl_2.wavelength = 705
        self.hsd.calib_wavelength_data = [wl_1, wl_2]

        gv_hints = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform | \
                   QPainter.RenderHint.TextAntialiasing

        self.ui_slit_angle_graphics_view.setScene(self.slit_angle_graphics_scene)
        self.ui_slit_angle_graphics_view.setRenderHints(gv_hints)
        self.ui_slit_angle_graphics_view.setMouseTracking(True)
        self.ui_slit_angle_graphics_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.ui_slit_angle_graphics_view.marquee_area_changed.connect(self.on_marquee_area_changed)

        self.ui_bdt_graphics_view.setScene(self.bdt_graphics_scene)
        self.ui_bdt_graphics_view.setRenderHints(gv_hints)
        self.ui_bdt_graphics_view.setMouseTracking(True)
        self.ui_bdt_graphics_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.ui_bdt_graphics_view.marquee_area_changed.connect(self.on_marquee_area_changed)

        self.ui_wt_graphics_view.setScene(self.wt_graphics_scene)
        self.ui_wt_graphics_view.setRenderHints(gv_hints)
        self.ui_wt_graphics_view.setMouseTracking(True)
        self.ui_wt_graphics_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self.ui_wt_graphics_view.marquee_area_changed.connect(self.on_marquee_area_changed)

        dashed_pen = QPen(QColor("white"))
        dashed_pen.setStyle(Qt.PenStyle.DashLine)
        dashed_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        dashed_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        dashed_pen.setWidth(2)

        dashed_pen_marquee = QPen(dashed_pen)

        dashed_pen_slit = QPen(dashed_pen_marquee)
        dashed_pen_slit.setColor(QColor("red"))

        dashed_pen_slit_roi_rect = QPen(dashed_pen_marquee)
        dashed_pen_slit_roi_rect.setColor(QColor("#cc870f"))

        dashed_pen_wavelength_line = QPen(dashed_pen_marquee)
        dashed_pen_wavelength_line.setColor(QColor("green"))

        brush_marquee = QBrush(QColor(255, 255, 255, 128))
        brush_marquee.setStyle(Qt.BrushStyle.BDiagPattern)

        brush_spectrum = QBrush(QColor(255, 0, 0, 192))
        brush_spectrum.setStyle(Qt.BrushStyle.BDiagPattern)

        circle_pen = QPen(QColor("red"))
        circle_pen.setWidth(2)

        polygon_pen = QPen(QColor("white"))
        polygon_pen.setWidth(2)

        polygon_brush = QBrush(QColor("red"))
        polygon_brush.setStyle(Qt.BrushStyle.SolidPattern)

        self.slit_graphics_line_item.setPen(dashed_pen_slit)
        self.slit_graphics_line_item.setOpacity(0.5)

        self.slit_graphics_roi_rect_item.setPen(dashed_pen_slit_roi_rect)
        self.slit_graphics_roi_rect_item.setOpacity(0.5)

        self.slit_graphics_marquee_area_rect_item.setPen(dashed_pen_marquee)
        self.slit_graphics_marquee_area_rect_item.setBrush(brush_marquee)
        self.slit_graphics_marquee_area_rect_item.setOpacity(0.5)

        self.bdt_graphics_marquee_area_rect_item.setPen(dashed_pen_marquee)
        self.bdt_graphics_marquee_area_rect_item.setBrush(brush_marquee)
        self.bdt_graphics_marquee_area_rect_item.setOpacity(0.5)

        self.bdt_graphics_center_x_line_item.setPen(dashed_pen_slit)
        self.bdt_graphics_center_x_line_item.setOpacity(0.25)

        self.bdt_graphics_center_y_line_item.setPen(dashed_pen_slit)
        self.bdt_graphics_center_y_line_item.setOpacity(0.25)

        self.wt_graphics_slit_line_item.setPen(dashed_pen_slit)
        self.wt_graphics_slit_line_item.setOpacity(0.5)

        self.wt_graphics_wavelength_line_item.setPen(dashed_pen_wavelength_line)
        self.wt_graphics_wavelength_line_item.setOpacity(0.25)

        self.ui_slit_image_threshold_value_checkbox.setEnabled(False)
        self.ui_slit_image_threshold_value_horizontal_slider.setEnabled(False)
        self.ui_slit_image_threshold_value_spinbox.setEnabled(False)
        self.ui_slit_angle_horizontal_slider.setEnabled(False)
        self.ui_slit_angle_double_spinbox.setEnabled(False)
        self.ui_slit_intercept_horizontal_slider.setEnabled(False)
        self.ui_slit_intercept_double_spinbox.setEnabled(False)
        self.ui_calc_slit_angle_button.setEnabled(False)

        self.ui_bdt_apply_rotation_checkbox.setEnabled(False)
        self.ui_bdt_slit_image_contrast_value_checkbox.setEnabled(False)
        self.ui_bdt_slit_image_contrast_value_spinbox.setEnabled(False)
        self.ui_bdt_slit_image_contrast_value_horizontal_slider.setEnabled(False)
        self.ui_bdt_distortion_grid_checkbox.setEnabled(False)
        self.ui_bdt_get_slit_center_button.setEnabled(False)
        self.ui_bdt_undistort_image_button.setEnabled(False)

        self.ui_bdew_center_x_spinbox.setEnabled(False)
        self.ui_bdew_center_y_spinbox.setEnabled(False)
        self.ui_bdew_grid_tile_size_spinbox.setEnabled(False)
        self.ui_bdew_equation_table_view_model = EquationParamsTableModel()
        self.ui_bdew_equation_table_view.setModel(self.ui_bdew_equation_table_view_model)
        self.ui_bdew_equation_table_view.setMouseTracking(True)
        self.ui_bdew_equation_header_view_horizontal = \
            EquationParamsTableHeaderViewHorizontal(Qt.Orientation.Horizontal, self.ui_bdew_equation_table_view)
        self.ui_bdew_equation_header_view_horizontal.setHighlightSections(True)
        self.ui_bdew_equation_table_view.setHorizontalHeader(self.ui_bdew_equation_header_view_horizontal)
        self.ui_bdew_equation_header_view_vertical = \
            EquationParamsTableHeaderViewVertical(Qt.Orientation.Vertical, self.ui_bdew_equation_table_view)
        self.ui_bdew_equation_header_view_vertical.setProperty('id', 'checkable')
        self.ui_bdew_equation_header_view_vertical.setHighlightSections(True)
        checkbox_stylesheet = hsd_gui_utils.parse_qss_by_class_name(self.stylesheet, 'QCheckBox')
        self.ui_bdew_equation_header_view_vertical.checkbox_stylesheet = checkbox_stylesheet
        self.ui_bdew_equation_table_view.setVerticalHeader(self.ui_bdew_equation_header_view_vertical)
        self.ui_bdew_equation_header_view_vertical.checked_section_count_changed.connect(
            self.on_ui_bdew_equation_params_changed, Qt.ConnectionType.QueuedConnection)
        self.ui_bdew_equation_table_view_model.dataChanged[QModelIndex, QModelIndex, "QList<int>"].connect(
            self.on_ui_bdew_equation_table_view_data_changed)
        self.fill_bdew()

        self.ui_wt_current_wavelength_image_horizontal_slider.setEnabled(False)
        self.ui_wt_current_wavelength_image_spinbox.setEnabled(False)
        self.ui_wt_apply_rotation_checkbox.setEnabled(False)
        self.ui_wt_apply_undistortion_checkbox.setEnabled(False)
        self.ui_wt_apply_contrast_preview_checkbox.setEnabled(False)
        self.ui_wt_contrast_preview_value_horizontal_slider.setEnabled(False)
        self.ui_wt_contrast_preview_value_spinbox.setEnabled(False)

        self.ui_wcdw_wavelength_table_view_model = WavelengthCalibrationTableModel()
        self.ui_wcdw_wavelength_table_view.setModel(self.ui_wcdw_wavelength_table_view_model)
        self.ui_wcdw_wavelength_table_view.setMouseTracking(True)
        self.ui_wcdw_wavelength_table_view.horizontalHeader().setMinimumHeight(22)
        self.ui_wcdw_wavelength_table_view.horizontalHeader().resizeSection(0, 210)
        self.ui_wcdw_wavelength_table_view.horizontalHeader().setStretchLastSection(True)
        self.ui_wcdw_wavelength_table_view.setAlternatingRowColors(True)
        self.fill_wcdw()

    def fill_bdew(self):
        self.ui_bdew_equation_header_view_vertical.clear_data()
        poly_deg = self.ui_bdew_polynomial_degree_spinbox.value()
        ept_model = self.ui_bdew_equation_table_view.model()
        ept_model.insertRows(0, poly_deg + 1)

        self.ui_bdew_equation_table_view.horizontalHeader().setMinimumHeight(22)
        self.ui_bdew_equation_table_view.horizontalHeader().resizeSection(0, 210)
        self.ui_bdew_equation_table_view.horizontalHeader().setStretchLastSection(True)
        self.ui_bdew_equation_table_view.setAlternatingRowColors(True)
        self.check_ui_bdt_undistort_image_button_availability()

    def load_barrel_distortion_params(self):
        barrel_distortion_params = self.hsd.get_barrel_distortion_params()
        if barrel_distortion_params is not None:
            ept_model: EquationParamsTableModel = self.ui_bdew_equation_table_view.model()
            params = [barrel_distortion_params['powers'], barrel_distortion_params['coeffs'],
                      barrel_distortion_params['factors']]
            poly_deg = self.ui_bdew_polynomial_degree_spinbox.value()
            ept_model.load_data_from_list(params, poly_deg + 1)

            for idx in barrel_distortion_params['powers']:
                self.ui_bdew_equation_header_view_vertical.set_section_checked(idx, True)
            if barrel_distortion_params['center'] is not None:
                center_xy = barrel_distortion_params['center']
                if len(center_xy) == 2:
                    center_x, center_y = center_xy
                    if center_x > 0 and center_y > 0:
                        self.ui_bdew_center_x_spinbox.setValue(center_x)
                        self.ui_bdew_center_y_spinbox.setValue(center_y)
        self.ui_bdew_center_x_spinbox.setEnabled(True)
        self.ui_bdew_center_y_spinbox.setEnabled(True)
        self.ui_bdew_grid_tile_size_spinbox.setEnabled(True)

    def initialize_texts(self):
        text_font = QFont("Century Gothic", 20, QFont.Weight.Light)
        text_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

        self.slit_graphics_text_item.setDefaultTextColor(QColor("white"))
        self.slit_graphics_text_item.setFont(text_font)
        self.slit_graphics_text_item.setOpacity(0.5)

    def fill_device_type_combobox(self):
        d = HSDeviceType.to_dict()
        for k, v in d.items():
            self.ui_device_type_combobox.addItem(k, v)

    def fill_recent_devices_menu(self):
        for action in self.ui_recent_device_settings_action_list:
            self.ui_recent_devices_menu.removeAction(action)
            self.ui_recent_device_settings_action_list.clear()
        for path in self.recent_device_settings_path_list:
            action = QAction(path, self.ui_recent_devices_menu)
            self.ui_recent_devices_menu.addAction(action)
            self.ui_recent_device_settings_action_list.append(action)

        for action in self.ui_recent_device_settings_action_list:
            self.recent_device_settings_action_triggered_signal_mapper.setMapping(action, action.text())
            action.triggered.connect(self.recent_device_settings_action_triggered_signal_mapper.map)
        self.recent_device_settings_action_triggered_signal_mapper.mappedString.connect(
            self.on_ui_recent_device_settings_action_triggered)

    def fill_wcdw(self):
        wt_model = self.ui_wcdw_wavelength_table_view.model()
        wt_model.insertRows(0, 3)

    def initialize_settings_dict(self):
        settings_dict = {
            "program": "HSDeviceGUI",
            "generation_date": utils.current_date(),
            "recent_device_settings_path_list": self.recent_device_settings_path_list,
            "last_device_settings_path": self.device_settings_path,
        }
        return settings_dict

    def save_settings(self):
        self.recent_device_settings_path_list = list(set(self.recent_device_settings_path_list))
        self.settings_dict["generation_date"] = utils.current_date()
        self.settings_dict["generation_time"] = utils.current_time()
        self.settings_dict["recent_device_settings_path_list"] = self.recent_device_settings_path_list
        self.settings_dict["last_device_settings_path"] = self.last_device_settings_path

        with open(self.settings_name, 'w') as settings_file:
            json.dump(self.settings_dict, settings_file, indent=4)

    def load_settings(self):
        settings_filename = self.settings_name
        if utils.path_exists(settings_filename):
            with open(settings_filename) as settings_file:
                self.settings_dict = json.load(settings_file)
            # Settings tab
            if utils.key_exists_in_dict(self.settings_dict, "recent_device_settings_path_list"):
                self.recent_device_settings_path_list = self.settings_dict["recent_device_settings_path_list"]
                self.fill_recent_devices_menu()

    def save_device_settings(self):
        self.device_settings_dict["generation_date"] = utils.current_date()
        self.device_settings_dict["generation_time"] = utils.current_time()
        self.device_settings_dict["slit_image_path"] = self.slit_image_path
        self.device_settings_dict["slit_threshold_value"] = self.ui_slit_image_threshold_value_spinbox.value()
        self.device_settings_dict["bdt_contrast_value"] = self.ui_bdt_slit_image_contrast_value_spinbox.value()
        self.device_settings_dict["bdt_grid_tile_size"] = self.ui_bdew_grid_tile_size_spinbox.value()
        self.device_settings_dict["wt_wavelength_image_dir_path"] = self.wt_wavelength_image_dir_path
        self.device_settings_dict["wt_contrast_value"] = self.ui_wt_contrast_preview_value_spinbox.value()
        self.device_settings_dict["device_metadata"] = self.hsd.to_dict()
        utils.save_dict_to_json(self.device_settings_dict, self.device_settings_path)

    def load_device_settings(self):
        if utils.path_exists(self.device_settings_path):
            self.device_settings_dict = utils.load_dict_from_json(self.device_settings_path)
            if utils.key_exists_in_dict(self.device_settings_dict, "slit_image_path"):
                self.slit_image_path = self.device_settings_dict["slit_image_path"]
                self.ui_slit_image_path_line_edit.setText(self.slit_image_path)
            else:
                self.slit_image_path = ""
                self.ui_slit_image_path_line_edit.setText(self.slit_image_path)
            if utils.key_exists_in_dict(self.device_settings_dict, "slit_threshold_value"):
                slit_threshold_value = self.device_settings_dict["slit_threshold_value"]
                self.ui_slit_image_threshold_value_horizontal_slider.setValue(slit_threshold_value)
                self.ui_slit_image_threshold_value_spinbox.setValue(slit_threshold_value)
            if utils.key_exists_in_dict(self.device_settings_dict, "bdt_contrast_value"):
                bdt_contrast_value = self.device_settings_dict["bdt_contrast_value"]
                self.ui_bdt_slit_image_contrast_value_horizontal_slider.setValue(bdt_contrast_value)
                self.ui_bdt_slit_image_contrast_value_spinbox.setValue(bdt_contrast_value)
                self.hsd.set_bd_contrast_value(bdt_contrast_value)
            if utils.key_exists_in_dict(self.device_settings_dict, "bdt_grid_tile_size"):
                bdt_grid_tile_size = self.device_settings_dict["bdt_grid_tile_size"]
                self.ui_bdew_grid_tile_size_spinbox.setValue(bdt_grid_tile_size)
                self.hsd.set_grid_tile_size(bdt_grid_tile_size)
            if utils.key_exists_in_dict(self.device_settings_dict, "wt_wavelength_image_dir_path"):
                self.wt_wavelength_image_dir_path = self.device_settings_dict["wt_wavelength_image_dir_path"]
                self.ui_wt_image_dir_path_line_edit.setText(self.wt_wavelength_image_dir_path)
                self.read_wl_image_dir.emit(self.wt_wavelength_image_dir_path)
            if utils.key_exists_in_dict(self.device_settings_dict, "wt_contrast_value"):
                wt_contrast_value = self.device_settings_dict["wt_contrast_value"]
                self.hsd.set_wl_contrast_value(wt_contrast_value)
                self.ui_wt_contrast_preview_value_horizontal_slider.setValue(wt_contrast_value)
                self.ui_wt_contrast_preview_value_spinbox.setValue(wt_contrast_value)
            if utils.key_exists_in_dict(self.device_settings_dict, "device_metadata"):
                device_data_dict = self.device_settings_dict["device_metadata"]
                self.hsd.load_dict(device_data_dict)
            self.apply_device_settings()

    def apply_device_settings(self):
        # TODO add mutex locker
        self.init_after_load_device_settings = True
        self.on_ui_load_slit_image_button_clicked()
        self.load_barrel_distortion_params()

    @staticmethod
    def flush_graphics_scene_data(graphics_scene: QGraphicsScene):
        for item in graphics_scene.items():
            graphics_scene.removeItem(item)

    @staticmethod
    def clear_marquee_area(graphics_rect_item: QGraphicsRectItem):
        graphics_rect_item.setRect(QRectF())

    @pyqtSlot(QImage)
    def receive_slit_image(self, slit_image_qt: QImage):
        self.slit_image_qt = slit_image_qt.copy()
        self.bdt_slit_image_qt = slit_image_qt.copy()
        # Slit angle tab graphics scene
        self.slit_angle_graphics_scene.removeItem(self.slit_graphics_text_item)
        self.slit_angle_graphics_scene.removeItem(self.slit_graphics_pixmap_item)
        self.clear_marquee_area(self.slit_graphics_marquee_area_rect_item)
        self.slit_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.slit_image_qt))
        self.slit_angle_graphics_scene.addItem(self.slit_graphics_pixmap_item)
        self.ui_slit_image_threshold_value_checkbox.setEnabled(True)
        self.ui_slit_image_threshold_value_horizontal_slider.setEnabled(True)
        self.ui_slit_image_threshold_value_spinbox.setEnabled(True)
        self.ui_calc_slit_angle_button.setEnabled(False)
        # Barrel distortion tab graphics scene
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_pixmap_item)
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.slit_image_qt))
        self.bdt_graphics_scene.addItem(self.bdt_graphics_pixmap_item)
        self.ui_bdt_apply_rotation_checkbox.setEnabled(True)
        self.ui_bdt_slit_image_contrast_value_checkbox.setEnabled(True)
        self.ui_bdt_slit_image_contrast_value_spinbox.setEnabled(True)
        self.ui_bdt_slit_image_contrast_value_horizontal_slider.setEnabled(True)

        # TODO add mutex locker
        if self.init_after_load_device_settings:
            self.on_compute_slit_angle_finished()
            self.ui_bdt_apply_rotation_checkbox.setChecked(True)
            self.on_ui_bdt_apply_rotation_checkbox_clicked(True)
            self.draw_bd_slit_data()

    @pyqtSlot(QImage)
    def receive_slit_preview_image(self, image_qt: QImage):
        self.slit_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot()
    def on_compute_slit_angle_finished(self):
        self.ui_slit_angle_horizontal_slider.setValue(int(self.hsd.get_slit_angle() * self.slit_angle_slider_mult))
        self.ui_slit_angle_double_spinbox.setValue(self.hsd.get_slit_angle())
        self.ui_slit_intercept_horizontal_slider.setValue(self.hsd.get_slit_intercept(to_int=True))
        self.ui_slit_intercept_double_spinbox.setValue(self.hsd.get_slit_intercept(to_int=True))
        self.ui_slit_angle_horizontal_slider.setEnabled(True)
        self.ui_slit_angle_double_spinbox.setEnabled(True)
        self.ui_slit_intercept_horizontal_slider.setEnabled(True)
        self.ui_slit_intercept_double_spinbox.setEnabled(True)

        self.draw_slit_data()

    @pyqtSlot(float, float)
    def on_adjust_slit_angle_range(self, range_min: float, range_max: float):
        # Don't touch slider while it has focus
        if not self.ui_slit_angle_horizontal_slider.hasFocus():
            self.ui_slit_angle_horizontal_slider.setRange(int(range_min * self.slit_angle_slider_mult),
                                                          int(range_max * self.slit_angle_slider_mult))

    @pyqtSlot(float, float)
    def on_adjust_slit_intercept_range(self, range_min: float, range_max: float):
        self.ui_slit_intercept_horizontal_slider.setRange(int(range_min), int(range_max))
        self.ui_slit_intercept_double_spinbox.setRange(int(range_min), int(range_max))

    def on_main_window_is_shown(self):
        self.load_settings()

    # Main window slots

    @pyqtSlot()
    def on_ui_file_open_action_triggered(self):
        file_path, _filter = QFileDialog.getOpenFileName(self, "Choose file", "",
                                                         "Metadata file (*.json)")
        if file_path != "":
            self.device_settings_path = file_path
            self.ui_device_settings_path_line_edit.setText(self.device_settings_path)
            self.last_device_settings_path = self.device_settings_path
            self.recent_device_settings_path_list.append(self.last_device_settings_path)
            self.recent_device_settings_path_list = list(set(self.recent_device_settings_path_list))
            self.fill_recent_devices_menu()
            self.load_device_settings()

    @pyqtSlot()
    def on_ui_file_exit_action_triggered(self):
        self.close()

    @pyqtSlot(str)
    def on_ui_recent_device_settings_action_triggered(self, path: str):
        if utils.path_exists(path):
            self.device_settings_path = path
            self.ui_device_settings_path_line_edit.setText(self.device_settings_path)
            self.last_device_settings_path = self.device_settings_path
            self.load_device_settings()
        else:
            for action in self.ui_recent_device_settings_action_list:
                if action.text() == path:
                    self.ui_recent_devices_menu.removeAction(action)
                    self.ui_recent_device_settings_action_list.remove(action)
                    self.recent_device_settings_path_list.remove(path)
                    self.show_message_box('Device settings',
                                          f"Can't find device settings by path:\n\"{path}\"!", 'warn')
                    break

    # Tab 0: slit angle tab slots

    @pyqtSlot(bool)
    def on_ui_slit_image_threshold_value_checkbox_clicked(self, checked: bool):
        if checked:
            self.threshold_slit_image.emit()
        else:
            self.slit_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.slit_image_qt))

    @pyqtSlot(int)
    def on_ui_slit_image_threshold_value_horizontal_slider_value_changed(self, value: int):
        self.hsd.threshold_value = value
        self.ui_slit_image_threshold_value_spinbox.setValue(value)

        if self.ui_slit_image_threshold_value_checkbox.isChecked():
            self.threshold_slit_image.emit()

    @pyqtSlot(int)
    def on_ui_slit_image_threshold_value_spinbox_value_changed(self, value: int):
        self.hsd.threshold_value = value
        self.ui_slit_image_threshold_value_horizontal_slider.setValue(value)

        if self.ui_slit_image_threshold_value_checkbox.isChecked():
            self.threshold_slit_image.emit()

    @pyqtSlot(int)
    def on_ui_slit_angle_horizontal_slider_value_changed(self, value: int):
        angle = value / self.slit_angle_slider_mult
        # Don't set angle while slider has no focus
        if self.ui_slit_angle_horizontal_slider.hasFocus():
            self.hsd.set_slit_angle(angle)
        self.ui_slit_angle_double_spinbox.setValue(angle)
        self.draw_slit_data()

    @pyqtSlot(float)
    def on_ui_slit_angle_double_spinbox_value_changed(self, value: float):
        # Don't set angle while spinbox has no focus
        if self.ui_slit_angle_double_spinbox.hasFocus():
            self.hsd.set_slit_angle(value)
        self.ui_slit_angle_horizontal_slider.setValue(int(value * self.slit_angle_slider_mult))
        self.draw_slit_data()

    @pyqtSlot(int)
    def on_ui_slit_intercept_horizontal_slider_value_changed(self, value: int):
        # Don't set intercept while slider has no focus
        if self.ui_slit_intercept_horizontal_slider.hasFocus():
            self.hsd.set_slit_intercept(value)
        self.ui_slit_intercept_double_spinbox.setValue(value)
        self.draw_slit_data()

    @pyqtSlot(float)
    def on_ui_slit_intercept_double_spinbox_value_changed(self, value: float):
        # Don't set intercept while spinbox has no focus
        if self.ui_slit_intercept_double_spinbox.hasFocus():
            self.hsd.set_slit_intercept(value)
        self.ui_slit_intercept_horizontal_slider.setValue(int(value))
        self.draw_slit_data()

    @pyqtSlot()
    def on_ui_slit_image_path_button_clicked(self):
        file_path, _filter = QFileDialog.getOpenFileName(self, "Choose file", "",
                                                         "Image file (*.bmp *.png *.jpg *.tif)")
        if file_path != "":
            self.slit_image_path = file_path
            self.ui_slit_image_path_line_edit.setText(self.slit_image_path)

    @pyqtSlot()
    def on_ui_load_slit_image_button_clicked(self):
        self.flush_graphics_scene_data(self.slit_angle_graphics_scene)
        self.read_slit_image.emit(self.slit_image_path)

    @pyqtSlot()
    def on_ui_calc_slit_angle_button_clicked(self):
        self.compute_slit_angle.emit(self.slit_graphics_marquee_area_rect_item.rect())

    # Tab 1: barrel distortion tab slots

    @pyqtSlot(bool)
    def on_ui_bdt_apply_rotation_checkbox_clicked(self, checked: bool):
        if checked:
            self.rotate_bd_slit_image.emit()
        else:
            self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.slit_image_qt))
            self.ui_bdt_slit_image_contrast_value_checkbox.setChecked(False)

    @pyqtSlot(bool)
    def on_ui_bdt_slit_image_contrast_value_checkbox_clicked(self, checked: bool):
        if checked:
            self.contrast_bd_slit_image.emit()
        else:
            image_qt = self.bdt_slit_image_qt
            if self.ui_bdt_apply_rotation_checkbox.isChecked():
                image_qt = self.bdt_slit_image_rotated_qt
            self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot(int)
    def on_ui_bdt_slit_image_contrast_value_horizontal_slider_value_changed(self, value: int):
        self.hsd.set_bd_contrast_value(value)
        self.ui_bdt_slit_image_contrast_value_spinbox.setValue(value)

        if self.ui_bdt_slit_image_contrast_value_checkbox.isChecked():
            self.contrast_bd_slit_image.emit()

    @pyqtSlot(int)
    def on_ui_bdt_slit_image_contrast_value_spinbox_value_changed(self, value: int):
        self.hsd.set_bd_contrast_value(value)
        self.ui_bdt_slit_image_contrast_value_spinbox.setValue(value)

        if self.ui_bdt_slit_image_contrast_value_checkbox.isChecked():
            self.contrast_bd_slit_image.emit()

    @pyqtSlot(bool)
    def on_ui_bdt_distortion_grid_checkbox_clicked(self, checked: bool):
        if checked:
            self.redraw_distortion_grid()
        else:
            image_qt = self.bdt_slit_image_qt
            if self.ui_bdt_slit_image_contrast_value_checkbox.isChecked():
                image_qt = self.bdt_slit_image_contrasted_qt
            elif self.ui_bdt_apply_rotation_checkbox.isChecked():
                image_qt = self.bdt_slit_image_rotated_qt
            if image_qt is not None:
                self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot()
    def on_ui_bdt_undistort_image_button_clicked(self):
        self.undistort_slit_image.emit(self.ui_bdt_slit_image_contrast_value_checkbox.isChecked())

    @pyqtSlot(QImage)
    def on_receive_bd_slit_image_rotated(self, image_qt: QImage):
        self.bdt_slit_image_rotated_qt = image_qt.copy()
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.bdt_slit_image_rotated_qt))

    @pyqtSlot(QImage)
    def on_receive_bd_slit_image_contrasted(self, image_qt: QImage):
        self.bdt_slit_image_contrasted_qt = image_qt.copy()
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.bdt_slit_image_contrasted_qt))

    @pyqtSlot(QImage)
    def on_receive_bd_distortion_grid_image(self, image_qt: QImage):
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))
        self.check_ui_bdt_undistort_image_button_availability()

    @pyqtSlot(QImage)
    def on_receive_bd_undistorted_slit_image(self, image_qt: QImage):
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))
        self.ui_bdt_distortion_grid_checkbox.setChecked(False)

    @pyqtSlot()
    def on_ui_bdt_equation_set_button_clicked(self):
        if self.bdew.isVisible() and not self.wcdw.hasFocus():
            self.bdew.activateWindow()
        else:
            self.bdew.show()

    @pyqtSlot()
    def on_ui_bdt_get_slit_center_button_clicked(self):
        threshold_value = self.ui_slit_image_threshold_value_spinbox.value()
        self.compute_bd_slit_center.emit(self.bdt_graphics_marquee_area_rect_item.rect(), threshold_value)

    @pyqtSlot(int, int)
    def on_receive_bd_slit_center(self, center_x: int, center_y: int):
        self.ui_bdew_center_x_spinbox.setValue(center_x)
        self.ui_bdew_center_y_spinbox.setValue(center_y)
        self.draw_bd_slit_data()

    @pyqtSlot()
    def on_ui_bdew_center_x_spinbox_editing_finished(self):
        center_x = self.ui_bdew_center_x_spinbox.value()
        center_y = self.ui_bdew_center_y_spinbox.value()
        if center_x > 0 and center_y > 0:
            self.hsd.set_center(center_x, center_y)
            self.draw_bd_slit_data()
            self.redraw_distortion_grid()

    @pyqtSlot()
    def on_ui_bdew_center_y_spinbox_editing_finished(self):
        center_x = self.ui_bdew_center_x_spinbox.value()
        center_y = self.ui_bdew_center_y_spinbox.value()
        if center_x > 0 and center_y > 0:
            self.hsd.set_center(center_x, center_y)
            self.draw_bd_slit_data()
            self.redraw_distortion_grid()

    @pyqtSlot()
    def on_ui_bdew_grid_tile_size_spinbox_editing_finished(self):
        value = self.ui_bdew_grid_tile_size_spinbox.value()
        self.hsd.set_grid_tile_size(value)
        self.redraw_distortion_grid()

    @pyqtSlot()
    def on_ui_bdew_polynomial_degree_spinbox_editing_finished(self):
        self.ui_bdt_distortion_grid_checkbox.setChecked(False)
        self.on_ui_bdt_distortion_grid_checkbox_clicked(False)
        poly_deg = self.ui_bdew_polynomial_degree_spinbox.value()
        if self.ui_bdew_equation_table_view.model().rowCount() != poly_deg + 1:
            self.ui_bdew_equation_table_view.model().clear()
            self.fill_bdew()

    @pyqtSlot(QModelIndex, QModelIndex, "QList<int>")
    def on_ui_bdew_equation_table_view_data_changed(self, top_left, bottom_right, roles):
        self.on_ui_bdew_equation_params_changed()

    @pyqtSlot()
    def on_ui_bdew_equation_params_changed(self):
        check_list = self.ui_bdew_equation_header_view_vertical.get_check_list()
        equation_params = {'center': [], 'powers': [], 'coeffs': [], 'factors': []}
        model: EquationParamsTableModel = self.ui_bdew_equation_table_view.model()
        for i in range(len(check_list)):
            if check_list[i]:
                coeff = float(model.data(model.index(i, 0), Qt.ItemDataRole.DisplayRole))
                factor = float(model.data(model.index(i, 1), Qt.ItemDataRole.DisplayRole))
                equation_params['powers'].append(i)
                equation_params['coeffs'].append(coeff)
                equation_params['factors'].append(factor)
        if len(equation_params['powers']) > 0:
            ep_dev = self.hsd.get_barrel_distortion_params()
            if ep_dev is not None:
                equation_params['center'] = ep_dev['center']
            self.hsd.set_barrel_distortion_params(equation_params)
            if self.ui_bdt_distortion_grid_checkbox.isChecked():
                self.draw_distortion_grid.emit(self.ui_bdt_slit_image_contrast_value_checkbox.isChecked())
        self.check_ui_bdt_undistort_image_button_availability()

    # Tab 2: wavelengths tab slots

    @pyqtSlot()
    def on_ui_wt_image_dir_path_open_button_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose directory", self.wt_wavelength_image_dir_path)
        if dir_path != "":
            self.wt_wavelength_image_dir_path = dir_path
            self.ui_wt_image_dir_path_line_edit.setText(self.wt_wavelength_image_dir_path)
            self.read_wl_image_dir.emit(self.wt_wavelength_image_dir_path)

    @pyqtSlot(int)
    def on_ui_wt_current_wavelength_image_horizontal_slider_value_changed(self, value: int):
        self.ui_wt_current_wavelength_image_spinbox.setValue(value)
        self.read_wl_image.emit(value,
                                self.ui_wt_apply_rotation_checkbox.isChecked(),
                                self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                self.ui_wt_apply_contrast_preview_checkbox.isChecked())

    @pyqtSlot(int)
    def on_ui_wt_current_wavelength_image_spinbox_value_changed(self, value: int):
        self.ui_wt_current_wavelength_image_horizontal_slider.setValue(value)
        self.read_wl_image.emit(value,
                                self.ui_wt_apply_rotation_checkbox.isChecked(),
                                self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                self.ui_wt_apply_contrast_preview_checkbox.isChecked())

    @pyqtSlot(bool)
    def on_ui_wt_apply_rotation_checkbox_clicked(self, checked: bool):
        self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                checked,
                                self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                self.ui_wt_apply_contrast_preview_checkbox.isChecked())
        self.draw_wl_data()

    @pyqtSlot(bool)
    def on_ui_wt_apply_undistortion_checkbox_clicked(self, checked: bool):
        self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                self.ui_wt_apply_rotation_checkbox.isChecked(),
                                checked,
                                self.ui_wt_apply_contrast_preview_checkbox.isChecked())

    @pyqtSlot(bool)
    def on_ui_wt_apply_contrast_preview_checkbox_clicked(self, checked: bool):
        self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                self.ui_wt_apply_rotation_checkbox.isChecked(),
                                self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                checked)

    @pyqtSlot(int)
    def on_ui_wt_contrast_preview_value_horizontal_slider_value_changed(self, value: int):
        self.hsd.set_wl_contrast_value(value)
        self.ui_wt_contrast_preview_value_spinbox.setValue(value)
        if self.ui_wt_apply_contrast_preview_checkbox.isChecked():
            self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                    self.ui_wt_apply_rotation_checkbox.isChecked(),
                                    self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                    True)

    @pyqtSlot(int)
    def on_ui_wt_contrast_preview_value_spinbox_value_changed(self, value: int):
        self.hsd.set_wl_contrast_value(value)
        self.ui_wt_contrast_preview_value_horizontal_slider.setValue(value)
        if self.ui_wt_apply_contrast_preview_checkbox.isChecked():
            self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                    self.ui_wt_apply_rotation_checkbox.isChecked(),
                                    self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                    True)

    @pyqtSlot()
    def on_ui_wt_wavelength_calibration_data_window_show_button_clicked(self):
        if self.wcdw.isVisible() and not self.wcdw.hasFocus():
            self.wcdw.activateWindow()
        else:
            self.wcdw.show()

    @pyqtSlot(int)
    def on_receive_wl_image_count(self, value: int):
        self.ui_wt_current_wavelength_image_horizontal_slider.setMaximum(value)
        self.ui_wt_current_wavelength_image_spinbox.setMaximum(value)
        self.ui_wt_current_wavelength_image_horizontal_slider.setValue(0)
        self.ui_wt_current_wavelength_image_spinbox.setValue(0)
        self.ui_wt_current_wavelength_image_horizontal_slider.setEnabled(True)
        self.ui_wt_current_wavelength_image_spinbox.setEnabled(True)
        self.ui_wt_apply_rotation_checkbox.setEnabled(True)
        self.ui_wt_apply_undistortion_checkbox.setEnabled(True)
        self.ui_wt_apply_contrast_preview_checkbox.setEnabled(True)
        self.ui_wt_contrast_preview_value_horizontal_slider.setEnabled(True)
        self.ui_wt_contrast_preview_value_spinbox.setEnabled(True)
        self.read_wl_image.emit(self.ui_wt_current_wavelength_image_spinbox.value(),
                                self.ui_wt_apply_rotation_checkbox.isChecked(),
                                self.ui_wt_apply_undistortion_checkbox.isChecked(),
                                self.ui_wt_apply_contrast_preview_checkbox.isChecked())

    @pyqtSlot(QImage)
    def on_receive_wl_image(self, wl_image_qt: QImage):
        self.wt_wavelength_image_qt = wl_image_qt.copy()
        self.wt_graphics_scene.removeItem(self.wt_graphics_pixmap_item)
        self.wt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.wt_wavelength_image_qt))
        self.wt_graphics_scene.addItem(self.wt_graphics_pixmap_item)
        self.draw_wl_data()

    # Tab 4: settings tab slots

    @pyqtSlot()
    def on_ui_device_settings_path_save_button_clicked(self):
        self.device_settings_path, _filter = QFileDialog.getSaveFileName(self, "Save file", self.device_settings_path,
                                                                         "Settings file (*.json)")

        if self.device_settings_path != "":
            self.ui_device_settings_path_line_edit.setText(self.device_settings_path)
            self.device_settings_name = utils.file_complete_name(self.device_settings_path)

    @pyqtSlot()
    def on_ui_device_settings_save_button_clicked(self):
        self.save_device_settings()
        self.last_device_settings_path = self.device_settings_path
        # TODO rewrite
        self.recent_device_settings_path_list.append(self.last_device_settings_path)
        self.recent_device_settings_path_list = list(set(self.recent_device_settings_path_list))
        self.fill_recent_devices_menu()

    @pyqtSlot(QPointF, QPointF)
    def on_marquee_area_changed(self, top_left_on_scene: QPointF, bottom_right_on_scene: QPointF):
        graphics_view: Optional[QGraphicsView, HSGraphicsView] = QObject.sender(self)

        marquee_area_rect = QRectF(top_left_on_scene, bottom_right_on_scene)
        marquee_area_graphics_rect_item: Optional[QGraphicsRectItem] = None

        if graphics_view == self.ui_slit_angle_graphics_view:
            marquee_area_graphics_rect_item = self.slit_graphics_marquee_area_rect_item
        elif graphics_view == self.ui_bdt_graphics_view:
            marquee_area_graphics_rect_item = self.bdt_graphics_marquee_area_rect_item

        if marquee_area_graphics_rect_item is not None:
            marquee_area_graphics_rect_item.setRect(marquee_area_rect)
            if graphics_view == self.ui_slit_angle_graphics_view:
                self.ui_calc_slit_angle_button.setEnabled(not marquee_area_rect.isEmpty())
            elif graphics_view == self.ui_bdt_graphics_view:
                self.ui_bdt_get_slit_center_button.setEnabled(not marquee_area_rect.isEmpty())
            if marquee_area_graphics_rect_item not in graphics_view.scene().items():
                graphics_view.scene().addItem(marquee_area_graphics_rect_item)
            graphics_view.update()

    def check_ui_bdt_undistort_image_button_availability(self):
        if self.hsd.is_equation_data_enough():
            self.ui_bdt_distortion_grid_checkbox.setEnabled(True)
            self.ui_bdt_undistort_image_button.setEnabled(True)
        else:
            self.ui_bdt_distortion_grid_checkbox.setEnabled(False)
            self.ui_bdt_undistort_image_button.setEnabled(False)

    def draw_slit_data(self):
        self.slit_angle_graphics_scene.removeItem(self.slit_graphics_marquee_area_rect_item)
        self.slit_angle_graphics_scene.removeItem(self.slit_graphics_line_item)
        self.slit_angle_graphics_scene.removeItem(self.slit_graphics_roi_rect_item)
        self.slit_graphics_line_item.setLine(
            QLineF(0, self.hsd.get_slit_intercept(), self.slit_image_qt.width(),
                   self.hsd.get_slit_slope() * self.slit_image_qt.width() + self.hsd.get_slit_intercept()))
        x, y, w, h = self.hsd.get_slit_roi()
        self.slit_graphics_roi_rect_item.setRect(QRectF(x, y, w, h))
        self.slit_angle_graphics_scene.addItem(self.slit_graphics_line_item)
        self.slit_angle_graphics_scene.addItem(self.slit_graphics_roi_rect_item)

    def draw_bd_slit_data(self):
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_marquee_area_rect_item)
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_center_x_line_item)
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_center_y_line_item)

        if self.hsd.is_center_defined():
            center_x = self.ui_bdew_center_x_spinbox.value()
            center_y = self.ui_bdew_center_y_spinbox.value()
            self.bdt_graphics_center_x_line_item.setLine(
                QLineF(2, center_y, self.bdt_graphics_pixmap_item.boundingRect().width() - 3, center_y))
            self.bdt_graphics_center_y_line_item.setLine(
                QLineF(center_x, 2, center_x, self.bdt_graphics_pixmap_item.boundingRect().height() - 3))
            self.bdt_graphics_scene.addItem(self.bdt_graphics_center_x_line_item)
            self.bdt_graphics_scene.addItem(self.bdt_graphics_center_y_line_item)
        elif not self.bdt_graphics_marquee_area_rect_item.rect().isEmpty():
            if not self.ui_bdt_slit_image_contrast_value_checkbox.isChecked() or \
                    not self.ui_bdt_distortion_grid_checkbox.isChecked():
                self.bdt_graphics_scene.addItem(self.bdt_graphics_marquee_area_rect_item)

    def draw_wl_data(self):
        self.wt_graphics_scene.removeItem(self.wt_graphics_slit_line_item)
        self.wt_graphics_scene.removeItem(self.wt_graphics_wavelength_line_item)
        if self.ui_wt_apply_rotation_checkbox.isChecked():
            self.wt_graphics_slit_line_item.setLine(
                QLineF(0, self.hsd.get_slit_intercept_rotated(),
                       self.slit_image_qt.width(),
                       self.hsd.get_slit_intercept_rotated()))
        else:
            self.wt_graphics_slit_line_item.setLine(
                QLineF(0, self.hsd.get_slit_intercept(), self.slit_image_qt.width(),
                       self.hsd.get_slit_slope() * self.slit_image_qt.width() + self.hsd.get_slit_intercept()))
        self.wt_graphics_scene.addItem(self.wt_graphics_slit_line_item)

    def redraw_distortion_grid(self):
        if self.ui_bdt_distortion_grid_checkbox.isChecked():
            self.draw_distortion_grid.emit(self.ui_bdt_slit_image_contrast_value_checkbox.isChecked())

    def eventFilter(self, obj, event):
        # if obj == self.ui_slit_angle_graphics_view:
        #     if event.type() == QEvent.Type.MouseButtonPress:
        #         if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
        #             if event.button() == Qt.MouseButton.LeftButton:
        #                 self.slit_angle_graphics_scene.removeItem(self.slit_graphics_rect_item)
        #                 relative_origin = self.ui_slit_angle_graphics_view.mapToScene(event.pos())
        #                 rect_qt = QRectF(relative_origin, QPointF(30, 20))
        #                 rect_qt = QRectF(relative_origin, relative_origin)
        #                 self.slit_graphics_rect_item.setRect(rect_qt)
        #                 self.slit_angle_graphics_scene.addItem(self.slit_graphics_rect_item)
        #                 print(relative_origin)
        #                 return True
        #     elif event.type() == QEvent.Type.MouseMove:
        #         if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
        #             if event.button() == Qt.MouseButton.LeftButton:
        #                 relative_origin = self.ui_slit_angle_graphics_view.mapToScene(event.pos())
        #                 print(self.ui_slit_angle_graphics_view.mapToScene(event.pos()))
        #                 rect_qt = self.slit_graphics_rect_item.rect()
        #                 rect_qt.setBottomRight(relative_origin)
        #                 self.slit_graphics_rect_item.setRect(rect_qt)
        #         print("mm")
        #     elif event.type() == QEvent.Type.MouseButtonRelease:
        #         if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
        #             if event.button() == Qt.MouseButton.LeftButton:
        #                 relative_origin = self.ui_slit_angle_graphics_view.mapToScene(event.pos())
        #                 rect_qt = self.slit_graphics_rect_item.rect()
        #                 rect_qt.setBottomRight(relative_origin)
        #                 self.slit_graphics_rect_item.setRect(rect_qt)
        #                 # self.slit_angle_graphics_scene.addItem(self.slit_graphics_rect_item)
        #                 print(rect_qt)
        #                 print(relative_origin)
        #                 return True

        return super(HSDeviceGUI, self).eventFilter(obj, event)

    def closeEvent(self, event):
        self.bdew.close()
        self.wcdw.close()
        self.t_hsd.exit()
        self.save_settings()
        event.accept()

    def showEvent(self, event):
        event.accept()
        # zero interval timer fires only when after all events in the queue are processed
        QTimer.singleShot(0, self.on_main_window_is_shown)

    def show_message_box(self, title: str, message: str, message_type: str = 'info'):
        message_box = QMessageBox(QMessageBox.Icon.NoIcon, title, message, QMessageBox.StandardButton.Ok, self)
        message_box.setModal(True)
        icon_size = 36
        icon_qsize = QSize(icon_size, icon_size)
        icon_path = 'icons:ud-info-square.svg'
        if message_type == 'info':
            icon_path = 'icons:ud-info-square.svg'
        elif message_type == 'warn':
            icon_path = 'icons:ud-exclamation-square.svg'
        elif message_type == 'error':
            icon_path = 'icons:ud-x-square.svg'
        message_box.setIconPixmap(QIcon(icon_path).pixmap(icon_qsize))
        message_box.show()


def main():
    # TODO remove before release
    hsd_gui_utils.compile_scss_into_qss("Resources/Dark.scss", "Resources/Dark.qss")

    QDir.addSearchPath('icons', './Resources/Images/')
    QDir.addSearchPath('icons_gen', './Resources/Images/Generated/')
    QDir.addSearchPath('resources', './Resources/')
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
    hsd_gui = HSDeviceGUI()
    hsd_gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
