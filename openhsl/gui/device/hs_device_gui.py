import copy
import ctypes
import json
import sys

import numpy as np
from PyQt6.QtCore import Qt, QDir, QEvent, QLineF, QObject, QPointF, QRect, QRectF, QSignalMapper, QSize, QThread, \
    QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QActionGroup, QBrush, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, \
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem, \
    QGraphicsTextItem, QGraphicsScene, QGraphicsView, QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar, QMessageBox, \
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QToolBar, QToolButton, QWidget
from PyQt6 import uic
from typing import Any, Dict, List, Optional
from openhsl.build.hs_device import HSDevice, HSDeviceType, HSCalibrationSlitData, HSCalibrationWavelengthData
from openhsl.gui.device.custom_controls import CheckableLatexHeaderView, HSGraphicsView
from openhsl.gui.device.hs_device_qt import HSDeviceQt
import openhsl.gui.device.utils as hsd_gui_utils
import openhsl.build.utils as utils


# noinspection PyTypeChecker
class HSDeviceGUI(QMainWindow):
    read_slit_image = pyqtSignal(str)
    threshold_slit_image = pyqtSignal()
    compute_slit_angle = pyqtSignal(QRectF)
    rotate_bd_slit_image = pyqtSignal()
    threshold_bd_slit_image = pyqtSignal()
    edge_bd_slit_image = pyqtSignal(QRectF)

    def __init__(self):
        super(HSDeviceGUI, self).__init__()
        uic.loadUi('hs_device_mainwindow.ui', self)
        self.setWindowIcon(QIcon("icons:openhsl.svg"))

        # Workaround for taskbar icon in Windows
        # See: https://stackoverflow.com/a/1552105
        openhsl_id = 'locus.openhsl.hs_device_gui.0.0.1'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(openhsl_id)

        self.bdew = QWidget()
        uic.loadUi('bd_equation_form.ui', self.bdew)
        self.bdew.setWindowIcon(QIcon("icons:openhsl.svg"))

        qss_path = "./Resources/Dark.qss"
        # qss_path = "./Resources/Grey.qss"
        self.stylesheet = ""

        with open(qss_path, 'r') as f:
            self.stylesheet = f.read()
            self.setStyleSheet(self.stylesheet)
            self.bdew.setStyleSheet(self.stylesheet)

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
        self.ui_bdt_slit_image_threshold_value_checkbox: QCheckBox = \
            self.findChild(QCheckBox, 'bdtSlitImageThresholdValue_checkBox')
        self.ui_bdt_slit_image_threshold_value_horizontal_slider: QSlider = \
            self.findChild(QSlider, 'bdtSlitImageThresholdValue_horizontalSlider')
        self.ui_bdt_slit_image_threshold_value_spinbox: QSpinBox = \
            self.findChild(QSpinBox, 'bdtSlitImageThresholdValue_spinBox')
        self.ui_bdt_spectrum_edges_checkbox: QCheckBox = self.findChild(QCheckBox, 'bdtSpectrumEdges_checkBox')
        self.ui_bdt_equation_view_label: QLabel = self.findChild(QLabel, 'bdtEquationView_label')
        self.ui_bdt_equation_set_button: QPushButton = self.findChild(QPushButton, 'bdtEquationSet_pushButton')
        self.ui_bdt_equation_estimate_button: QPushButton = \
            self.findChild(QPushButton, 'bdtEquationEstimate_pushButton')
        # BDT: Barrel distortion equation window
        self.ui_bdew_equation_table_widget: QTableWidget = self.bdew.findChild(QTableWidget, 'equation_tableWidget')
        self.ui_bdew_polynomial_degree_spinbox: QSpinBox = self.bdew.findChild(QSpinBox, 'polynomialDegree_spinBox')
        self.ui_bdew_equation_checkable_header_view: Optional[CheckableLatexHeaderView] = None
        # Wavelengths tab
        self.ui_wavelength_table_widget: QTableWidget = self.findChild(QTableWidget, 'wavelength_tableWidget')
        # Settings tab
        self.ui_device_type_combobox: QComboBox = self.findChild(QComboBox, "deviceType_comboBox")
        self.ui_device_settings_path_line_edit: QLineEdit = self.findChild(QLineEdit, "deviceSettingsPath_lineEdit")
        self.ui_device_settings_path_save_button: QPushButton = \
            self.findChild(QPushButton, 'deviceSettingsPathSave_pushButton')
        self.ui_device_settings_save_button: QPushButton = \
            self.findChild(QPushButton, 'deviceSettingsSave_pushButton')

        # Signal and slot connections
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
        self.ui_bdt_slit_image_threshold_value_checkbox.clicked.connect(
            self.on_ui_bdt_slit_image_threshold_value_checkbox_clicked)
        self.ui_bdt_slit_image_threshold_value_horizontal_slider.valueChanged.connect(
            self.on_ui_bdt_slit_image_threshold_value_horizontal_slider_value_changed)
        self.ui_bdt_slit_image_threshold_value_spinbox.valueChanged.connect(
            self.on_ui_bdt_slit_image_threshold_value_spinbox_value_changed)
        self.ui_bdt_spectrum_edges_checkbox.clicked.connect(self.on_ui_bdt_spectrum_edges_checkbox_clicked)
        self.rotate_bd_slit_image.connect(self.hsd.on_rotate_bd_slit_image, Qt.ConnectionType.QueuedConnection)
        self.threshold_bd_slit_image.connect(self.hsd.on_threshold_bd_slit_image, Qt.ConnectionType.QueuedConnection)
        self.edge_bd_slit_image.connect(self.hsd.on_edge_bd_slit_image, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_image_rotated.connect(
            self.on_receive_bd_slit_image_rotated, Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_image_thresholded.connect(self.on_receive_bd_slit_image_thresholded,
                                                        Qt.ConnectionType.QueuedConnection)
        self.hsd.send_bd_slit_image_edged.connect(self.on_receive_bd_slit_image_edged,
                                                  Qt.ConnectionType.QueuedConnection)
        self.ui_bdt_equation_set_button.clicked.connect(self.on_ui_bdt_equation_set_button_clicked)
        self.ui_bdew_polynomial_degree_spinbox.valueChanged.connect(
            self.on_ui_bdew_polynomial_degree_spinbox_value_changed)
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
        self.bdt_graphics_spectrum_polygon_item = QGraphicsPolygonItem()
        self.bdt_slit_image_qt: Optional[QImage] = None
        self.bdt_spectrum_corners: Optional[np.ndarray] = None

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

        self.prepare_ui()
        # self.installEventFilter(self)

    def prepare_ui(self):
        self.fill_device_type_combobox()
        self.fill_wavelength_table_widget()
        # TODO maybe add default zeros
        self.hsd.calib_slit_data = HSCalibrationSlitData()
        # TODO remove
        wl_1 = HSCalibrationWavelengthData()
        wl_1.wavelength = 415
        wl_2 = HSCalibrationWavelengthData()
        wl_2.wavelength = 705
        self.hsd.calib_wavelength_data = [wl_1, wl_2]
        self.ui_wavelength_table_widget.insertRow(0)

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

        self.bdt_graphics_spectrum_polygon_item.setPen(dashed_pen_slit)
        self.bdt_graphics_spectrum_polygon_item.setBrush(brush_spectrum)
        self.bdt_graphics_spectrum_polygon_item.setOpacity(0.75)

        self.ui_slit_image_threshold_value_checkbox.setEnabled(False)
        self.ui_calc_slit_angle_button.setEnabled(False)

        self.ui_bdt_apply_rotation_checkbox.setEnabled(False)
        self.ui_bdt_equation_estimate_button.setEnabled(False)

        self.bdew.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.ui_bdew_equation_table_widget.setMouseTracking(True)
        self.ui_bdew_equation_checkable_header_view = CheckableLatexHeaderView(Qt.Orientation.Vertical,
                                                                               self.ui_bdew_equation_table_widget)
        self.ui_bdew_equation_checkable_header_view.setProperty('id', 'checkable')
        self.ui_bdew_equation_checkable_header_view.setHighlightSections(True)
        checkbox_stylesheet = hsd_gui_utils.parse_qss_by_class_name(self.stylesheet, 'QCheckBox')
        self.ui_bdew_equation_checkable_header_view.checkbox_stylesheet = checkbox_stylesheet
        self.ui_bdew_equation_table_widget.setVerticalHeader(self.ui_bdew_equation_checkable_header_view)
        self.fill_bdew()

    def fill_bdew(self):
        self.ui_bdew_equation_checkable_header_view.clear_data()
        poly_deg = self.ui_bdew_polynomial_degree_spinbox.value()
        self.ui_bdew_equation_table_widget.setColumnCount(1)
        self.ui_bdew_equation_table_widget.setRowCount(poly_deg + 1)
        vhl = ["$1$"]

        for i in range(poly_deg + 1):
            vhl.append(f"$r^{{{i + 1}}}$")
            twi = QTableWidgetItem(f"{1}")
            twi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui_bdew_equation_table_widget.setItem(i, 0, twi)

        self.ui_bdew_equation_checkable_header_view.generate_latex_labels(vhl, 10, '#d0d0d0')
        self.ui_bdew_equation_table_widget.setVerticalHeaderLabels([''] * len(vhl))

        self.ui_bdew_equation_table_widget.setHorizontalHeaderLabels(['Coefficient'])
        self.ui_bdew_equation_table_widget.horizontalHeader().setMinimumHeight(22)
        self.ui_bdew_equation_table_widget.horizontalHeader().resizeSection(0, 200)
        self.ui_bdew_equation_table_widget.horizontalHeader().setStretchLastSection(True)
        self.ui_bdew_equation_table_widget.setAlternatingRowColors(True)

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

    def fill_wavelength_table_widget(self):
        header_labels = ["Wavelength", "Slit slope", "Slit angle", "Slit intercept",
                         "Wavelength ROI x", "Wavelength ROI y", "Wavelength ROI width", "Wavelength ROI height",
                         "Wavelength slit offset"]
        self.ui_wavelength_table_widget.setColumnCount(len(header_labels))
        self.ui_wavelength_table_widget.setHorizontalHeaderLabels(header_labels)
        self.ui_wavelength_table_widget.horizontalHeader().setStretchLastSection(True)
        self.ui_wavelength_table_widget.resizeColumnsToContents()

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
        if utils.path_exists(settings_filename):
            with open(settings_filename) as settings_file:
                self.settings_dict = json.load(settings_file)
            # Settings tab
            if utils.key_exists_in_dict(self.settings_dict, "recent_device_settings_path_list"):
                self.recent_device_settings_path_list = self.settings_dict["recent_device_settings_path_list"]
                self.fill_recent_devices_menu()

    def save_device_settings(self):
        self.device_settings_dict["generation_date"] = utils.get_current_date()
        self.device_settings_dict["generation_time"] = utils.get_current_time()
        self.device_settings_dict["slit_image_path"] = self.slit_image_path
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
            if utils.key_exists_in_dict(self.device_settings_dict, "device_metadata"):
                device_data_dict = self.device_settings_dict["device_metadata"]
                self.hsd.load_dict(device_data_dict)
            self.apply_device_settings()

    def apply_device_settings(self):
        # TODO add mutex locker
        self.init_after_load_device_settings = True
        self.on_ui_load_slit_image_button_clicked()

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
        self.ui_calc_slit_angle_button.setEnabled(False)
        # Barrel distortion tab graphics scene
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_pixmap_item)
        self.clear_marquee_area(self.bdt_graphics_marquee_area_rect_item)
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.slit_image_qt))
        self.bdt_graphics_scene.addItem(self.bdt_graphics_pixmap_item)
        self.ui_bdt_apply_rotation_checkbox.setEnabled(True)

        # TODO add mutex locker
        if self.init_after_load_device_settings:
            self.on_compute_slit_angle_finished()
            self.ui_bdt_apply_rotation_checkbox.setChecked(True)
            self.on_ui_bdt_apply_rotation_checkbox_clicked(True)

    @pyqtSlot(QImage)
    def receive_slit_preview_image(self, image_qt: QImage):
        self.slit_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot()
    def on_compute_slit_angle_finished(self):
        self.ui_slit_angle_horizontal_slider.setValue(int(self.hsd.get_slit_angle() * self.slit_angle_slider_mult))
        self.ui_slit_angle_double_spinbox.setValue(self.hsd.get_slit_angle())
        self.ui_slit_intercept_horizontal_slider.setValue(self.hsd.get_slit_intercept(to_int=True))
        self.ui_slit_intercept_double_spinbox.setValue(self.hsd.get_slit_intercept(to_int=True))

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

    @pyqtSlot(bool)
    def on_ui_bdt_slit_image_threshold_value_checkbox_clicked(self, checked: bool):
        if checked:
            self.threshold_bd_slit_image.emit()
        else:
            self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.bdt_slit_image_qt))

    @pyqtSlot(int)
    def on_ui_bdt_slit_image_threshold_value_horizontal_slider_value_changed(self, value: int):
        self.hsd.bd_threshold_value = value
        self.ui_bdt_slit_image_threshold_value_spinbox.setValue(value)

        if self.ui_bdt_slit_image_threshold_value_checkbox.isChecked():
            self.threshold_bd_slit_image.emit()
        elif self.ui_bdt_spectrum_edges_checkbox.isChecked():
            self.edge_bd_slit_image.emit(self.bdt_graphics_marquee_area_rect_item.rect())

    @pyqtSlot(int)
    def on_ui_bdt_slit_image_threshold_value_spinbox_value_changed(self, value: int):
        self.hsd.bd_threshold_value = value
        self.ui_bdt_slit_image_threshold_value_spinbox.setValue(value)

        if self.ui_bdt_slit_image_threshold_value_checkbox.isChecked():
            self.threshold_bd_slit_image.emit()
        elif self.ui_bdt_spectrum_edges_checkbox.isChecked():
            self.edge_bd_slit_image.emit(self.bdt_graphics_marquee_area_rect_item.rect())

    @pyqtSlot(bool)
    def on_ui_bdt_spectrum_edges_checkbox_clicked(self, checked: bool):
        if checked:
            self.edge_bd_slit_image.emit(self.bdt_graphics_marquee_area_rect_item.rect())
            self.ui_bdt_slit_image_threshold_value_checkbox.setChecked(False)
        else:
            self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(self.bdt_slit_image_qt))

    @pyqtSlot(QImage)
    def on_receive_bd_slit_image_rotated(self, image_qt: QImage):
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot(QImage)
    def on_receive_bd_slit_image_thresholded(self, image_qt: QImage):
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))

    @pyqtSlot(QImage, np.ndarray)
    def on_receive_bd_slit_image_edged(self, image_qt: QImage, corners: np.ndarray):
        self.bdt_graphics_pixmap_item.setPixmap(QPixmap.fromImage(image_qt))
        self.bdt_spectrum_corners = corners
        self.draw_bd_slit_data()

    @pyqtSlot()
    def on_ui_bdt_equation_set_button_clicked(self):
        self.bdew.show()

    @pyqtSlot(int)
    def on_ui_bdew_polynomial_degree_spinbox_value_changed(self, value: int):
        self.ui_bdew_equation_table_widget.clear()
        self.fill_bdew()

    # Tab 2: wavelengths tab slots

    # Tab 4: settings tab slots

    @pyqtSlot()
    def on_ui_device_settings_path_save_button_clicked(self):
        self.device_settings_path, _filter = QFileDialog.getSaveFileName(self, "Save file", self.device_settings_path,
                                                                         "Settings file (*.json)")

        if self.device_settings_path != "":
            self.ui_device_settings_path_line_edit.setText(self.device_settings_path)
            self.device_settings_name = utils.get_file_complete_name(self.device_settings_path)

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
            self.ui_calc_slit_angle_button.setEnabled(not marquee_area_rect.isEmpty())
            if marquee_area_graphics_rect_item not in graphics_view.scene().items():
                graphics_view.scene().addItem(marquee_area_graphics_rect_item)
            graphics_view.update()

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
        self.bdt_graphics_scene.removeItem(self.bdt_graphics_spectrum_polygon_item)

        if self.bdt_spectrum_corners is not None:
            polygon = QPolygonF()
            for p in self.bdt_spectrum_corners:
                polygon.append(QPointF(p[1], p[0]))
            self.bdt_graphics_spectrum_polygon_item.setPolygon(polygon)
            self.bdt_graphics_scene.addItem(self.bdt_graphics_spectrum_polygon_item)
        else:
            self.bdt_graphics_scene.addItem(self.bdt_graphics_marquee_area_rect_item)

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
    QDir.addSearchPath('resources', './Resources/')
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
    hsd_gui = HSDeviceGUI()
    hsd_gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
