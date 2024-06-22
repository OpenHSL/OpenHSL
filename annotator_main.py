# Import GUI specific items
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsView
import sys
import traceback
import os
import numpy as np
import cv2
from qimage2ndarray import array2qimage, recarray_view

from annotator.lib.tkmask import generate_tk_defects_layer
from annotator.lib.annotmask import get_sqround_mask  # New mask generation facility (original mask needed)

from annotator.ui_lib.QtImageAnnotator import QtImageAnnotator

# Specific UI features
from PyQt5.QtWidgets import QSplashScreen, QMessageBox, QGraphicsScene, QFileDialog, QTableWidgetItem, QDialog, QLabel, QPushButton, QVBoxLayout, QLineEdit, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt5.QtCore import Qt, QRectF, QSize


from annotator.ui import annotator_ui, color_specs_ui
import configparser
import time
import datetime
import subprocess

import pandas as pd

from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask
from openhsl.paint.utils import ANDVI, ANDI, cluster_hsi
from scipy.io import loadmat
import qimage2ndarray

from annotator.ui.palette import PaletteGrid, PaletteHorizontal, PaletteVertical

from functools import partial    

# Overall constants
PUBLISHER = "HSI"
APP_TITLE = "Annotator"
APP_VERSION = "v.1.3"

# Some configs
BRUSH_DIAMETER_MIN = 1
BRUSH_DIAMETER_MAX = 80
BRUSH_DIAMETER_DEFAULT = 1


# Colors
MARK_COLOR_MASK = QColor(255,0,0,99)
MARK_COLOR_DEFECT_DEFAULT = QColor(0, 0, 255, 99)
HELPER_COLOR = QColor(0,0,0,99)

# Some paths
COLOR_DEF_PATH = "../annotator/defs/color_defs.csv"


# Color definitions window
class AnnotatorGUIColorSpec(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(AnnotatorGUIColorSpec, self).__init__(parent)
        self.ui = color_specs_ui.Ui_ColorSpecsUI()
        self.ui.setupUi(self)


# Main UI class with all methods
class AnnotatorGUI(QtWidgets.QMainWindow, annotator_ui.Ui_AnnotatorMainWindow):
    # Applications states in status bar
    APP_STATUS_STATES = {"ready": "Ready.",
                         "loading": "Loading image...",
                         "exporting_layers": "Exporting layers...",
                         "no_images": "No images or unexpected folder structure."}

    # Annotation modes
    ANNOTATION_MODE_MARKING_DEFECTS = 0
    ANNOTATION_MODE_MARKING_MASK = 1

    ANNOTATION_MODES_BUTTON_TEXT = {ANNOTATION_MODE_MARKING_DEFECTS: "Hide layers",
                                    ANNOTATION_MODE_MARKING_MASK: "Hide layers"}
    ANNOTATION_MODES_BUTTON_COLORS = {ANNOTATION_MODE_MARKING_DEFECTS: "blue",
                                      ANNOTATION_MODE_MARKING_MASK: "red"}

    # Mask file extension. If it changes in the future, it is easier to swap it here
    MASK_FILE_EXTENSION_PATTERN = ".mask.png"

    # Config file
    config_path = None  # Path to config file
    config_data = None  # The actual configuration
    CONFIG_NAME = "annotator_config.ini"  # Name of the config file

    has_image = None
    img_shape = None

    # Flag which tells whether images were found in CWD
    dir_has_images = False

    # Drawing mode
    annotation_mode = ANNOTATION_MODE_MARKING_DEFECTS

    # Annotator
    annotator = None

    # Brush
    brush = None
    brush_diameter = BRUSH_DIAMETER_DEFAULT

    # Color definitions
    cspec = None

    # For TK
    tk_colors = None

    current_paint = None  # Paint of the brush

    # Color conversion dicts
    d_rgb2gray = None
    d_gray2rgb = None
    d_gray2rgb_arr = []

    # Immutable items
    current_image = None  # Original image
    current_mask = None  # Original mask
    current_helper = None  # Helper mask
    current_tk = None  # Defects mareked by TK

    # User-updatable items
    current_mask = None  # Defects mask
    current_updated_mask = None  # Updated mask

    # Image nameA paint device can only be painted by one painter at a time.
    current_img = None
    current_img_as_listed = None

    # Internal vars
    initializing = False
    app = None
    
    #
    HSI_SLIDER_MIN = 0
    HSI_SLIDER_MAX = 102
    HSI_SLIDER_DEFAULT = 40
    img_name = ""
    img_name_no_ext = ""
    img_path = ""
    
    isANDImethod = False
    
    colors_arr =(['#8bc92e', '#2825cc', '#e6f542', '#9c0202','#2ed9d6', '#bf1f82', '#fa6e02', '#00ff2f', '#ff7dc7', '#e04f4f']) # , '#6d8fc9'
    colors_gray_arr =([10, 25, 50, 60, 75, 85, 100, 125, 150, 175]) 


    def __init__(self, parent=None):

        self.initializing = True

        # Setting up the base UI
        super(AnnotatorGUI, self).__init__(parent)
        self.setupUi(self)


        self.annotator = QtImageAnnotator()

        # Need to synchronize brush sizes with the annotator
        self.annotator.MIN_BRUSH_DIAMETER = BRUSH_DIAMETER_MIN
        self.annotator.MAX_BRUSH_DIAMETER = BRUSH_DIAMETER_MAX
        self.annotator.brush_diameter = BRUSH_DIAMETER_DEFAULT

        self.figThinFigure.addWidget(self.annotator)

        # Config file storage: config file stored in user directory
        self.config_path = self.fix_path(os.path.expanduser("~")) + "." + PUBLISHER + os.sep

        # Get color specifications and populate the corresponding combobox
        self.read_defect_color_defs()
        self.add_colors_to_list()

        # Assign necessary dicts in the annotator component
        if self.d_rgb2gray is not None and self.d_gray2rgb is not None:
            self.annotator.d_rgb2gray = self.d_rgb2gray
            self.annotator.d_gray2rgb = self.d_gray2rgb
        #else:
        #    raise RuntimeError("Failed to load the color conversion schemes. Annotations cannot be saved.")

        # Set up second window
        self.color_ui = AnnotatorGUIColorSpec(self)

        # Update button states
        self.update_button_states()

        # Initialize everything
        self.initialize_brush_slider()
        self.initialize_HSI_slider()

        # Log this anyway
        self.log("Application started")

        # Style the mode button properly
        self.annotation_mode_default()

        # Initialization completed
        self.initializing = False

        # Set up the status bar
        self.status_bar_message("ready")
        
        self.directory = ""        
        self.loaded_hsi = np.empty([0, 0, 0])
        self.wavelengths = []
        self.loaded_hsmask = np.empty([0, 0, 0])
        self.label_class = []      
        self.mask_layers = 0
        self.hsi = None
        
        self.hsmask = HSMask()
        self.hsi = HSImage()
        self.key_answer = ""
        self.mask_all_colors = []

    # Set up those UI elements that depend on config
    def UI_config(self):
        # Check whether log should be shown or not
        self.check_show_log()

        # TODO: TEMP: For buttons, use .clicked.connect(self.*), for menu actions .triggered.connect(self.*),
        # TODO: TEMP: for checkboxes use .stateChanged, and for spinners .valueChanged
        self.actionLog.triggered.connect(self.update_show_log)
        #self.actionColor_definitions.triggered.connect(self.open_color_definition_help)
        self.actionProcess_original_mask.triggered.connect(self.process_mask)
        self.actionSave_current_annotations.triggered.connect(self.save_masks)
        
        #lara
        #self.actionLoad.triggered.connect(self.browse_load_mask_directory)

        # Reload AI-generated mask, if present in the directory
        self.actionAIMask.triggered.connect(self.load_AI_mask)

        # Button assignment
        self.annotator.mouseWheelRotated.connect(self.accept_brush_diameter_change)
        self.btnClear.clicked.connect(self.clear_all_annotations)
        self.InvertingMask.clicked.connect(self.invert_mask)
        self.btnBrowseImageDir.clicked.connect(self.browse_image_directory)
        self.makemask.clicked.connect(self.load_AI_mask)
      
        self.pushButton_2.clicked.connect(self.save_masks)
        
        self.add_layer.clicked.connect(self.add_layerclass_to_mask)
        self.delete_layer.clicked.connect(self.delete_layerclass)
        self.btnMode.clicked.connect(self.annotation_mode_switch)
        self.actionLoadmask.clicked.connect(self.browse_load_mask_directory)
        # Selecting new image from list
        # NB! We depend on this firing on index change, so we remove manual load_image elsewhere
            #self.connect_image_load_on_list_index_change(True)

        # Try to load an image now that everything is initialized
        #self.load_image()

    def open_color_definition_help(self):

        if not self.cspec:
            self.log("Cannot show color specifications as none are loaded")
            return

        # Assuming colors specs were updated, set up the table
        ###lara###
        self.b_apply = self.color_ui.ui.change_layer_data
        self.b_apply.clicked.connect(self.change_layer_data_func)
        
        self.b_add_layer = self.color_ui.ui.add_new_layer_pb
        self.b_add_layer.clicked.connect(self.add_new_layer_pb_func)      
        ###lara###         
        self.t = self.color_ui.ui.tabColorSpecs
        t=self.t
        t.setRowCount(len(self.cspec))
        t.setColumnCount(4) #3
        t.setColumnWidth(0, 100) # 0, 150
        t.setColumnWidth(1, 100) # 1, 150
        t.setColumnWidth(2, 100) # 2, 150
        t.setColumnWidth(3, 200) 
        t.setHorizontalHeaderLabels(["Colors", "Colors (trans)", "Load Layer"])

        self.pb_note_array = []
        # Go through the color specifications and set them appropriately
        row = 0
        for col in self.cspec:
            tk = col["COLOR_HEXRGB_TK"]
            nus = col["COLOR_NAME_EN"]
            net = col["COLOR_NAME_ET"]
            dt = col["COLOR_HEXRGB"]
            gr = col["COLOR_GSCALE_MAPPING"]

            # Text
            t.setItem(row, 0, QTableWidgetItem(net))
            t.setItem(row, 1, QTableWidgetItem(nus))
            t.setItem(row, 2, QTableWidgetItem(str(gr)))
            
            ###lara### кнопки добавления готового слоя       
            pb_note = QtWidgets.QPushButton(t)
            pb_note.clicked.connect(lambda: self.add_full_layer(row))
            pb_note.setText(". . .")
            t.setCellWidget(row,3,pb_note)
            self.pb_note_array.append(pb_note)
            ###lara###
            
            # Background and foreground
            t.item(row, 0).setBackground(QColor(tk))
            t.item(row, 0).setForeground(self.get_best_fg_for_bg(QColor(tk)))

            t.item(row, 1).setBackground(QColor(dt))
            t.item(row, 1).setForeground(self.get_best_fg_for_bg(QColor(dt)))

            t.item(row, 2).setBackground(QColor(gr, gr, gr))
            t.item(row, 2).setForeground(self.get_best_fg_for_bg(QColor(gr, gr, gr)))

            row += 1

        self.color_ui.show()

###lara###
    def add_new_layer_pb_func(self):  
        
        t = self.t
        row= t.rowCount()
        row = row + 1
        self.pb_note_array = []        
        t.setRowCount(row)       

        tk = "COLOR_HEXRGB_TK"
        nus = "COLOR_NAME_EN"
        net = self.cspec[0]["COLOR_NAME_ET"]
        print(net, row, "add_new_layer net, row")
        dt = "COLOR_HEXRGB"
        gr = "COLOR_GSCALE_MAPPING"
        bt = "BUTTON_ADD"

        # Text
        t.setItem(row, 0, QTableWidgetItem(net))
        t.setItem(row, 1, QTableWidgetItem(nus))
        t.setItem(row, 2, QTableWidgetItem(str(gr)))            
        # кнопки добавления готового слоя       
        pb_note = QtWidgets.QPushButton(t)
        pb_note.clicked.connect(lambda: self.add_full_layer(row))
        pb_note.setText(". . .")
        t.setCellWidget(row,3,pb_note)      
        self.pb_note_array.append(pb_note)
        #######
        # Background and foreground
        self.add_colors_to_list()    
              
###lara###
    def change_layer_data_func(self):
               
        nb_row = len(self.cspec)
        nb_col = 2

        for row in range (nb_row):
            #for col in range(nb_col):
            new_row_data = self.color_ui.ui.tabColorSpecs.item(row, 0).text()
            #self.cspec[row] = new_row_data
            self.cspec[row]['COLOR_NAME_ET'] = new_row_data
          
        print(self.cspec, "Change_layer_data self.cspec")
        self.add_colors_to_list()

    '''
    def connect_image_load_on_list_index_change(self, state):
        if state:
            self.lstImages.currentIndexChanged.connect(self.load_image)
        else:
            self.lstImages.disconnect()
    '''

    def initialize_brush_slider(self):
        self.sldBrushDiameter.setMinimum(BRUSH_DIAMETER_MIN)
        self.sldBrushDiameter.setMaximum(BRUSH_DIAMETER_MAX)
        self.sldBrushDiameter.setValue(BRUSH_DIAMETER_DEFAULT)
        self.sldBrushDiameter.valueChanged.connect(self.brush_slider_update)
        self.brush_slider_update()

    def brush_slider_update(self):
        new_diameter = self.sldBrushDiameter.value()
        self.txtBrushDiameter.setText(str(new_diameter))
        self.brush_diameter = new_diameter
        self.update_annotator()
        
    def initialize_HSI_slider(self):
        self.HSI_slider.setMinimum(self.HSI_SLIDER_MIN)
        self.HSI_slider.setMaximum(self.HSI_SLIDER_MAX)
        self.HSI_slider.setValue(self.HSI_SLIDER_DEFAULT)
        self.HSI_slider.valueChanged.connect(self.HSI_slider_update) #  self.HSI_slider_update (self.load_image)        
        #self.HSI_slider_update()
        # выдергивать по индексу из self.wavelengths[1] каждый слой
    def HSI_slider_update(self):
        new_HSI_slider_VALUE = self.HSI_slider.value() # self.HSI_slider.value()
        self.txt_HSI_slider.setText(str(self.wavelengths[self.HSI_slider.value()]))
        self.txt_HSI_slider_leyer.setText(str(self.HSI_slider.value()))
        
        self.new_HSI_slider_VALUE = new_HSI_slider_VALUE
        # UPDATE HSI QIMAGE
        #self.clear_all_annotations()                

        self.load_layer_hsi_image()    
        self.update_annotator_view()    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        #self.update_annotator()        

    def accept_brush_diameter_change(self, change):

        # Need to disconnect slider while changing value
        self.sldBrushDiameter.valueChanged.disconnect()

        new_diameter = int(self.sldBrushDiameter.value()+change)
        new_diameter = BRUSH_DIAMETER_MIN if new_diameter < BRUSH_DIAMETER_MIN else new_diameter
        new_diameter = BRUSH_DIAMETER_MAX if new_diameter > BRUSH_DIAMETER_MAX else new_diameter
        self.sldBrushDiameter.setValue(new_diameter)
        self.txtBrushDiameter.setText(str(new_diameter))

        # Reconnect to slider move interrupt
        self.sldBrushDiameter.valueChanged.connect(self.brush_slider_update)

    def invert_mask(self):
        print(self.current_mask, "current_mask")
        self.current_mask = np.abs(self.current_mask - 1)
        self.setmasktocolor(self.current_mask, is_multy_layer = False)
        
     
    # Clear currently used paint completely
    def clear_all_annotations(self):
        h, w = self.current_image.rect().height(), self.current_image.rect().width()
        self.img_shape = (h, w)
        print(self.img_shape, "self.img_shape")
        img_new = np.zeros(self.img_shape, dtype=np.uint8)
        
        if self.annotation_mode is self.ANNOTATION_MODE_MARKING_DEFECTS:
            self.current_mask = img_new
        elif self.annotation_mode is self.ANNOTATION_MODE_MARKING_MASK:
            self.current_updated_mask = 255-img_new  
                 
        self.annotator.setFocus()
        self.update_annotator_view(is_for_clear=True)
        

    def update_annotator(self):
        if self.annotator is not None:
            self.annotator.brush_diameter = self.brush_diameter
            self.annotator.update_brush_diameter(0)
            self.annotator.brush_fill_color = self.current_paint

    def update_mask_from_current_mode(self):
        the_mask = self.get_updated_mask()
        if self.annotation_mode is self.ANNOTATION_MODE_MARKING_DEFECTS:
            self.current_mask = the_mask
        else:
            self.current_updated_mask = the_mask

    # Change annotation mode
    def annotation_mode_switch(self):

        # Save the mask
        #self.update_mask_from_current_mode()
        
        # Update the UI
        self.annotation_mode += 1
        
        if self.annotation_mode > 1:
            self.annotation_mode = 0
        self.current_paint = [MARK_COLOR_DEFECT_DEFAULT, MARK_COLOR_MASK][self.annotation_mode]
        if self.annotation_mode == self.ANNOTATION_MODE_MARKING_DEFECTS:  # TODO: this should be optimized
            self.change_brush_color()
            self.lstDefectsAndColors.setEnabled(True)
        else:
            self.lstDefectsAndColors.setEnabled(False)
        self.update_annotator()
        self.btnMode.setText(self.ANNOTATION_MODES_BUTTON_TEXT[self.annotation_mode])
        self.btnMode.setStyleSheet("QPushButton {font-weight: bold; color: "
                                   + self.ANNOTATION_MODES_BUTTON_COLORS[self.annotation_mode] + "}")
        
        # Update the view
        self.update_annotator_view()
        self.annotator.setFocus()

    # Set default annotation mode
    def annotation_mode_default(self):
        self.annotation_mode = self.ANNOTATION_MODE_MARKING_DEFECTS
        self.current_paint = [MARK_COLOR_DEFECT_DEFAULT, MARK_COLOR_MASK][self.annotation_mode]
        if self.annotation_mode == self.ANNOTATION_MODE_MARKING_DEFECTS:
            self.change_brush_color()
            self.lstDefectsAndColors.setEnabled(True)
        else:
            self.lstDefectsAndColors.setEnabled(False)
        self.update_annotator()
        self.btnMode.setText(self.ANNOTATION_MODES_BUTTON_TEXT[self.annotation_mode])
        self.btnMode.setStyleSheet("QPushButton {font-weight: bold; color: "
                                   + self.ANNOTATION_MODES_BUTTON_COLORS[self.annotation_mode] + "}")

    #def show_graph(self, method_name ):
        '''
        frame =QtWidgets.QFrame()
        self.setCentralWidget(frame)
        layout = QtWidgets.QHBoxLayout()
        frame.setLayout(layout)
        self.fileOpenButton_1 = QtWidgets.QPushButton('ANDVI',self)
        self.fileOpenButton_2 = QtWidgets.QPushButton('ANDI',self)
        self.fileOpenButton_3 = QtWidgets.QPushButton('cluster_hsi',self)
        layout.addWidget(self.fileOpenButton_1)
        layout.addWidget(self.fileOpenButton_2)
        layout.addWidget(self.fileOpenButton_3)
        self.fileOpenButton_1.clicked.connect(self.buttonClicked("ANDVI"))
        self.fileOpenButton_2.clicked.connect(self.buttonClicked("ANDI"))
        self.fileOpenButton_3.clicked.connect(self.buttonClicked("cluster_hsi"))        
        '''
        

    # Get both masks as separate numpy arrays
    def get_updated_mask(self):
        if self.annotator._overlayHandle is not None:

            # Depending on the mode, fill the mask appropriately
            # Marking defects
            if self.annotation_mode is self.ANNOTATION_MODE_MARKING_DEFECTS:
                self.status_bar_message("exporting_layers")
                self.log("Exporting color layers...")
                the_new_mask = self.annotator.export_rgb2gray_mask()  # Easy, as this is implemented in annotator
                #the_new_mask = 255 * np.ones(self.img_shape, dtype=np.uint8)
                self.status_bar_message("ready")

            # Or updating the road edge mask
            else:
                mask = self.annotator.export_ndarray_noalpha()
                the_new_mask = 255 * np.ones(self.img_shape, dtype=np.uint8)

                # NB! This approach beats np.where: it is 4.3 times faster!
                reds, greens, blues = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]

                # Set the mask according to the painted road mask
                m1 = list(MARK_COLOR_MASK.getRgb())[:-1]
                the_new_mask[(reds == m1[0]) & (greens == m1[1]) & (blues == m1[2])] = 0

            return the_new_mask
        
        
        
    def update_mask_view(self):
        self.annotator.clearAndSetMaskOnly(self.current_mask, # self.current_mask,
                                        helper = None, # array2qimage(helper),
                                        aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                        process_gray2rgb=False, # process_gray2rgb=True,
                                        direct_mask_paint=False) # direct_mask_paint=True)
                
        
        

    def update_annotator_view(self, is_for_clear = False):

        # If there is no image, there's nothing to clear
        if self.current_image is None:
            return

        if is_for_clear == True:
            #h, w = self.current_image.rect().height(), self.current_image.rect().width()
            #mask = 255 * np.zeros((h, w, 4), dtype=np.uint8)
            #mask[self.current_updated_mask == 0] = list(MARK_COLOR_MASK.getRgb())
            
            mask = self.current_mask# self.current_mask  self.current_updated_mask

            #self.current_updated_mask = 255-img_new
            print("-----  1   ---- update annotaotr view-------------")    
            self.annotator.update_MaskLayers(self.current_image, # self.current_image,
                            mask, # hsmask.data[:, :, 1] self.current_mask,
                            helper = None, # array2qimage(helper),
                            aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                            process_gray2rgb=True, # process_gray2rgb=True,
                            direct_mask_paint=False,
                            color = self.mask_all_colors,
                            is_not_cube_mask = True) # direct_mask_paint=True))     

           
        else: 
            mask = self.hsmask    
            print(" ------------------обнова каждую смену ползунка hsi / update annotaotr view -----------------------")
            if self.annotation_mode is self.ANNOTATION_MODE_MARKING_DEFECTS:
                h, w = self.current_image.rect().height(), self.current_image.rect().width()                            
                helper = np.zeros((h,w,4), dtype=np.uint8)
                helper[self.current_helper == 0] = list(HELPER_COLOR.getRgb())
                # self.current_image=qimage2ndarray.array2qimage(self.loaded_hsi[:, :, self.wavelengths.index(self.HSI_slider.value())])
               
                #print(mask.data)
                #if mask.data == None:
                #    print(type(mask))
                #    h, w = self.current_image.rect().height(), self.current_image.rect().width()                            
                #    mask = np.zeros((h,w,1), dtype=np.uint8)
                    
                self.annotator.update_MaskLayers(self.current_image, # self.current_image,
                                            mask, # hsmask.data[:, :, 1] self.current_mask,
                                            helper = None, # array2qimage(helper),
                                            aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                            process_gray2rgb=True, # process_gray2rgb=True,
                                            direct_mask_paint=False,
                                            color = self.mask_all_colors,
                                            is_hidden = False,
                                            data_shape = True) # direct_mask_paint=True))     
                
            else:
                print(" ------------------обнова при загрузки-----------------------")
            
                # Remember, the mask must be inverted here, but saved properly
                h, w = self.current_image.rect().height(), self.current_image.rect().width()
                mask = 255 * np.zeros((h, w, 1), dtype=np.uint8)
                mask[self.current_updated_mask == 0] = list(MARK_COLOR_MASK.getRgb())
                
                cind = self.lstDefectsAndColors.currentIndex()
                lol = []
                #a,b,c = self.hsmask.data.shape()
                lol = self.hsmask.data
                lol[:,:,cind] = mask[:,:,1]
                #self.hsmask.data.itemset(4, 0)
                
                print("-----  3   ---- update annotaotr view-------------") 
                self.annotator.update_MaskLayers(self.current_image, # self.current_image,
                            lol, # hsmask.data[:, :, 1] self.current_mask,
                            helper = None, # array2qimage(helper),
                            aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                            process_gray2rgb=True, # process_gray2rgb=True,
                            direct_mask_paint=False,
                            color = self.mask_all_colors,
                            is_hidden = True) # direct_mask_paint=True))     
         
                #self.annotator.clearAndSetImageAndMask(self.current_image, mask)
             

    def process_mask(self):

        # Check the state of the checkbox and save it
        proc_mask = '0'
        if self.actionProcess_original_mask.isChecked():
            proc_mask = '1'
        self.config_data['MenuOptions']['ProcessMask'] == proc_mask
        self.config_save()

        # Now, reload the image
        self.load_image()


    def load_layer_hsi_image(self):        
        self.status_bar_message("loading")
        self.current_image=qimage2ndarray.array2qimage(self.loaded_hsi[:, :, self.HSI_slider.value()]) # self.wavelengths.index(self.HSI_slider.value())
        print("load_layer_hsi_image !!!!")
        
        
    def load_base_mask(self):
        #self.status_bar_message("loading base mask")
        self.app.processEvents()
        img_name = self.img_name
        img_name_no_ext = self.img_name_no_ext
        img_path = self.img_path
        try:
            self.log("Drawing defect marks on original image...")
            warning = []
            img_tk = generate_tk_defects_layer(self.txtImageDir.text(), self.txtImageDir.text(), # self.txtImageDir.text(), self.txtShpDir.text(),
                                                img_name_no_ext, self.tk_colors, warning, log=self.log)
            self.current_tk = img_tk

            # Check if there were warnings
            #if warning:
            #    self.show_info_box("Issues drawing the TK layer",
            #                       "There was some issue while drawing the TK layer: " + "; ".join(warning) + " " +
            #                       "Please check the log for details.",
            #                       QMessageBox.Warning)

        except Exception as e:
            #self.show_info_box("Error drawing the TK layer",
            #                   "There was some issue while drawing the TK layer. Please check the log for details.",
            #                   QMessageBox.Warning)
            self.actionLoad_marked_image.setChecked(False)
            self.log("Could not find or load the shapefile data. Will load only the image.")
            self.log("Additional details about this error: (" + str(e.__class__.__name__) + ") " + str(e))
            self.curent_tk = None

        # Shape of the image
        h, w = self.current_image.rect().height(), self.current_image.rect().width()
        self.img_shape = (h, w)

        # Load the mask and generate the "helper" mask
        #try:
        #    self.current_mask = cv2.imread(img_path + self.MASK_FILE_EXTENSION_PATTERN, cv2.IMREAD_GRAYSCALE)
        #    self.current_helper = get_sqround_mask(self.current_mask)
        #except:
        #    print("Cannot find the mask file. Please make sure FILENAME.mask.png " +
        #            "files exist in the folder for every image")
        #    self.log("Cannot find the mask file. Please make sure FILENAME.mask.png files exist in the folder for every image")
        #    self.status_bar_message("no_images")
        #    return

        # Set also default annotation mode
        self.annotation_mode_default()

        # Mask v2 just contains a copy of the default mask
        img_m = self.current_mask.copy()

        
        # No defect marks by default
        img_d = np.zeros(self.img_shape, dtype=np.uint8)

        # Now we set up the mutable images. NB! They are not COPIES, but references here
        self.current_mask = img_d
        self.current_updated_mask = img_m

        # Once all that is done, we need to update the actual image working area
        self.update_annotator_view()

        # Need to set focus on the QGraphicsScene so that shortcuts would work immediately
        self.annotator.setFocus()

        self.status_bar_message("ready")
        self.log("Done loading image")
        
     
    # Loads the image
    def load_image(self):
        print(" def load_image() ")
            
            
    def convertToMaskLeyer(self, boolmask):
        boolmask = boolmask * 100
        cv2.imwrite("self.img_path" + "ai_mask.png", boolmask)
        
    def setmasktocolor(self, ai_mask, is_multy_layer = True, is_ai_n_clusters = False):
        
        #cind = self.lstDefectsAndColors.currentIndex()
        #color = self.cspec[cind]

        cind = len(self.cspec)
        if cind == -1:
            cind = 0                
        print(ai_mask.shape, "ai_mask")        
        
        #import matplotlib.pyplot as plt
        #plt.show(ai_mask)
        #plt.show()                
        
        if is_multy_layer== True:
            if is_ai_n_clusters == True:
                num_class = int(self.message_box_ok_edit.text())              
            self.apeend_new_lstDefectsAndColors(ai_mask, num_class, is_cluster = True)
        
        else: 

            # Create the necessary dicts
            g2rgb = {}
            rgb2g = {}
            tk2rgb = {}                             
            
            print(np.max(ai_mask), np.min(ai_mask), "max и min")
            
            ai_mask_int = ai_mask.astype(int)
            print(ai_mask_int, "ai_mask_int")
            
            k = self.lstDefectsAndColors.currentIndex()
            print(k, "setmaskcolor - k")
            
            if k == -1:                
                #rgb_val = self.colors_arr[k] 
                #g_val = self.cspec[k]["COLOR_GSCALE_MAPPING"]                
                rgb_val = self.colors_arr[0]  
                g_val = self.colors_gray_arr[0]
                color = self.colors_arr[0]
                
                the_color = QColor("#63" + color.split("#")[1])
                self.add_layer_current_color = self.colors_arr[0]                
                
                self.cspec.append({"NAME_LAYER_D": "New class", 
                    "COLOR_HEXRGB": self.colors_arr[0],
                    "COLOR_GSCALE_MAPPING": self.colors_gray_arr[0]})
            
                #for col, row in self.cspec:            
                #rgb_val = self.cspec[i]["COLOR_HEXRGB"]  
                #g_val = self.cspec[i]["COLOR_GSCALE_MAPPING"]    
                name_val = self.cspec[0]["NAME_LAYER_D"]             

                print("Selected: {}".format(self.colors_arr[0]), "Selected: (self.colors_arr[i])")

                pix = QPixmap(50, 50)
                pix.fill(QColor(rgb_val))
                ticon = QIcon(pix)                            
                self.lstDefectsAndColors.addItem(ticon, " " + name_val +
                    " | "  + rgb_val + " | "  + str(g_val))
                         
            else:
                rgb_val = self.cspec[k]["COLOR_HEXRGB"] 
                g_val = self.cspec[k]["COLOR_GSCALE_MAPPING"]
                color = self.colors_arr[k]
                the_color = QColor("#63" + color.split("#")[1])
                                                                        
                self.add_layer_current_color = self.colors_arr[k]
            
        
            #g_val = self.cspec[k]["NAME_LAYER_D"] # col["NAME_LAYER_D"]     
            print(rgb_val,g_val, "rgb_val,g_val")
            
            ####################################
            # Fill in necessary dicts
            g2rgb[g_val] = rgb_val
            rgb2g[rgb_val] = g_val        

            # Set up dicts
            self.d_rgb2gray = rgb2g
            self.d_gray2rgb = g2rgb
            self.tk_colors = tk2rgb
            
            self.annotator.d_rgb2gray = self.d_rgb2gray
            self.annotator.d_gray2rgb = self.d_gray2rgb
            ###################################
            print(color, "setmaskcolor - color")            
            self.current_paint = the_color
            self.annotator.brush_fill_color = the_color
            #self.current_image = self.hsi.data        
            self.annotator.clearAndSetMaskOnly(self.current_image, # self.current_image,
                                                    ai_mask_int, # self.current_mask,
                                                    helper = None, # array2qimage(helper),
                                                    aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                                    process_gray2rgb=True, # process_gray2rgb=True,
                                                    direct_mask_paint=False,
                                                    color = the_color) # direct_mask_paint=True))   
            
          
    def onClicked(self, btn):
        from functools import partial  
        if btn == 'load': # btn.text()
            #ai_mask =            
            self.current_mask = ai_mask
            self.setmasktocolor(ai_mask, is_multy_layer = False)        
        
        elif btn == 'ANDVI': # btn.text()
            ai_mask = ANDVI(self.hsi)            
            self.current_mask = ai_mask
            self.setmasktocolor(ai_mask, is_multy_layer = False)

            #self.convertToMaskLeyer(ai_mask)

        elif btn == 'ANDI':
            print("check")#ai_mask = ANDI(self.hsi)
            
            self.message_box_label.setText("Ввыделите 2 области HSI и нажмите еще раз ANDI")
            
            self.annotator.isANDImethod = True
            self.isANDImethod = True
        
            message_box_ok_button_4 = QPushButton("Продолжить")
            message_box_ok_button_4.clicked.connect(partial(self.onClicked, "gonext")) 
            self.message_box_layout.addWidget(message_box_ok_button_4)
            
            
        elif btn == 'gonext':            
            area_1 = self.hsi.data[self.annotator.min_x_1:self.annotator.max_x_1, self.annotator.min_y_1:self.annotator.max_y_1, :]
            area_2 = self.hsi.data[self.annotator.min_x_2:self.annotator.max_x_2, self.annotator.min_y_2:self.annotator.max_y_2, :]
            
            ai_mask = ANDI(self.hsi, area_1, area_2)  # ANDI(self.hsi, area_1, area_2).astype(int)  
            
            print(self.annotator.min_x_1, self.annotator.max_x_1, self.annotator.min_x_2, self.annotator.max_x_2)
            print(self.hsi.data)
            print(area_1, "area_1")
            print(area_2, "area_1")
            
            #print(ai_mask.shape, " ai_mask.shape ")
            ai_mask = ai_mask[:,:]
            self.current_mask = ai_mask
            #print(type(ai_mask))
            #print(ai_mask.shape)
            #print(len(ai_mask), "type(ai_mask), ai_mask.shape, len(ai_mask)")

            #import matplotlib.pyplot as plt
            #plt.plot(self.current_mask)
            #plt.show()
            
            self.setmasktocolor(self.current_mask, is_multy_layer = False) # ai_mask[:,:,0,0] self.current_mask,
            print(self.current_mask, "self.current_mask")

                       
            #self.addloadedlayermask(ai_mask)  
            #from PIL import Image
            #im = Image.fromarray(ai_mask)
            #im.save("D:/_AII/2023/!_HSI/_save data/your_file.jpeg")
            #cv2.imwrite(self.img_path, ai_mask)
            
        elif btn == 'cluster_hsi':    
            from functools import partial    
            self.message_box_dialog_2 = QDialog(self , Qt.Window | Qt.WindowStaysOnTopHint) #  self , Qt.Window | Qt.WindowStaysOnTopHint
            #message_box_dialog.setModal(True)
            self.message_box_dialog_2.setWindowTitle("Параметры генерации")
            self.message_box_dialog_2.resize(300, 150)

            self.message_box_label_2 = QLabel("Введите число класов")
            self.message_box_ok_edit = QLineEdit()
            
            self.message_box_label_3 = QLabel("Введите тип кластеризатора")
            self.message_box_ok_typeclass = QComboBox()
            self.message_box_ok_typeclass.addItems(['KMeans', 'SpectralClustering'])
            
            message_box_ok_button_complite = QPushButton("Complite")
            message_box_ok_button_complite.clicked.connect(partial(self.set_category)) 
            
            message_box_layout = QVBoxLayout()
            self.message_box_dialog_2.setLayout(message_box_layout)

            message_box_layout.addWidget(self.message_box_label_2)
            message_box_layout.addWidget(self.message_box_ok_edit)
            message_box_layout.addWidget(self.message_box_label_3)
            message_box_layout.addWidget(self.message_box_ok_typeclass)
            message_box_layout.addWidget(message_box_ok_button_complite)

            self.message_box_dialog_2.show()  
            self.message_box_dialog_2.exec_()           
               
               
        #img_d = cv2.imread(self.img_path + "ai_mask.png", cv2.IMREAD_GRAYSCALE)
        #self.current_mask = img_d
        #self.update_annotator_view()
        #self.log("Replaced the current defect mask with the automatically generated one.")
        
        
    def set_category(self):
        print(self.message_box_ok_edit.text(), "self.message_box_ok_edit.text()")
        num_class = int(self.message_box_ok_edit.text())
        ai_mask = cluster_hsi(self.hsi, n_clusters = num_class, cl_type = str(self.message_box_ok_typeclass.currentText()))          
        
        print(np.max(ai_mask), np.min(ai_mask), "max min")
        
        '''
        h, w = ai_mask.shape
        mask_ones = np.ones((h, w), np.uint8)
        new_ai_mask = ai_mask + mask_ones
        
        #new_ai_mask = [int(x == 1) for x in new_ai_mask]  
       
        tet =np.stack((ai_mask, new_ai_mask))
        print(tet, "tet")
        '''

        #for i in range(0, np.max(ai_mask)):
        #unique_values = np.unique(ai_mask)  # Получаем уникальные значения из массива AR
        
        arr = HSMask.convert_2d_to_3d_mask(ai_mask)
        
        '''
        sorted_AR = np.zeros_like(ai_mask)  # Создаем новый массив с нулями такого же размера, как и AR
        fin = []
        
        for value in range(0, np.max(ai_mask)): # unique_values
            sorted_AR[ai_mask == value] = 1  # Для каждого повторяющегося значения value в AR, изменяем его на 1
            print(sorted_AR, "ssssssssssssss")  # Выводим отсортированный массив
            fin.append(sorted_AR)
            arr = np.array(fin)
   
        #for l in fin:
        #    self.setmasktocolor(l, is_multy_layer = False, is_ai_n_clusters = False)
        
        
        h,w,g = arr.shape
        arr = np.transpose(arr, (1, 2, 0)) #np.array(arr).reshape(w, g, h)
        #fin = np.stack(fin)        
        self.current_mask = arr
        '''
        print(arr.shape, "ai_mask.shape keeeek")        
        
        self.setmasktocolor(arr, is_multy_layer = True, is_ai_n_clusters = True)   
        
            
    '''        
    def addloadedlayermask(self, mask):
        self.current_mask = qimage2ndarray.array2qimage(mask)
        self.current_mask = self.current_mask
        
        self.annotator.clearAndSetImageAndMask(self.current_image, # self.current_image,
                                                   self.current_mask, # self.current_mask,
                                                   helper = None, # array2qimage(helper),
                                                   aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                                   process_gray2rgb=False, # process_gray2rgb=True,
                                                   direct_mask_paint=False) # direct_mask_paint=True)
        
        #self.annotator.loadImageFromFile(self.current_mask)        
        self.update_annotator_view()         
    '''   
    
                                                            
    def load_AI_mask(self):        
       
        from functools import partial       
        self.message_box_dialog = QDialog(self , Qt.Window | Qt.WindowStaysOnTopHint) #  self , Qt.Window | Qt.WindowStaysOnTopHint
        #message_box_dialog.setModal(True)
        self.message_box_dialog.setWindowTitle("Внимание")
        self.message_box_dialog.resize(300, 150)


 
        self.message_box_label = QLabel("Выберите метод для получения маски")
        
        message_box_ok_button_0 = QPushButton("Load Binary")
        message_box_ok_button_0.clicked.connect(partial(self.onClicked, "load")) # message_box_ok_button.clicked.connect(self.onClicked("ANDVI")) 
     
        message_box_ok_button = QPushButton("ANDVI метод")
        message_box_ok_button.clicked.connect(partial(self.onClicked, "ANDVI")) # message_box_ok_button.clicked.connect(self.onClicked("ANDVI")) 
        
        message_box_ok_button_2 = QPushButton("ANDI метод")
        message_box_ok_button_2.clicked.connect(partial(self.onClicked, "ANDI")) 
        
        message_box_ok_button_3 = QPushButton("Класстеризация")
        message_box_ok_button_3.clicked.connect(partial(self.onClicked, "cluster_hsi")) 
        
        self.message_box_layout = QVBoxLayout()
        self.message_box_dialog.setLayout(self.message_box_layout)

        self.message_box_layout.addWidget(self.message_box_label)
        self.message_box_layout.addWidget(message_box_ok_button_0)
        self.message_box_layout.addWidget(message_box_ok_button)
        self.message_box_layout.addWidget(message_box_ok_button_2)
        self.message_box_layout.addWidget(message_box_ok_button_3)

        
        self.message_box_dialog.show()  
        self.message_box_dialog.exec_()


    def save_masks(self):

        # Update the current mask
        #self.update_mask_from_current_mode()
        
        file_name = ""
        default_dir ="/home/"
        default_filename = os.path.join(default_dir, file_name)
        files_types = "mat (*.mat);;h5 (*.h5);;tiff (*.tiff);;npy (*.npy);;PNG img (*.png);; BMP img (*.bmp)" # "mat File (*.mat *.h5 *.tiff *.npy)" "h5 File (*.mat *.h5 *.tiff *.npy)"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить HSI маску", default_filename, files_types
        )
        if filename:
            print(filename)
            save_dir = filename
            save_path_defects = filename
            save_path_masks = filename
        
        #save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите")    
        #save_path_defects = save_dir + "/mask_01.png" # save_path_defects = save_dir + self.current_img + ".defect.mask.png"
        #save_path_masks = save_dir + "/mask_02.png" # save_path_masks = save_dir + self.current_img + ".cut.mask_v2.png"
        ####       

        #self.hsmask.data = self.current_mask
        print(self.hsi.data, "save_masks - self.hsi.data")
        self.hsmask.save(save_path_masks, self.key_answer)
        #self.hsmask.save_to_mat(save_path_masks, self.key_answer)
        ####


    # In-GUI console log
    def log(self, line):
        # Get the time stamp
        ts = datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S] ')
        self.txtConsole.moveCursor(QtGui.QTextCursor.End)
        self.txtConsole.insertPlainText(ts + line + os.linesep)

        # Only do this if app is already referenced in the GUI (TODO: a more elegant solution?)
        if self.app is not None:
            self.app.processEvents()

    def check_show_log(self):
        if self.actionLog.isChecked():
            self.gbApplicationLog.show()
        else:
            self.gbApplicationLog.hide()

    def update_show_log(self):
        self.check_show_log()
        self.store_menu_options_to_config()

    @staticmethod
    def open_file_in_os(fn):
        if sys.platform == "win32":
            os.startfile(fn)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, fn])

    # Path related functions
    @staticmethod
    def fix_path(p):
        # Only if string is nonempty
        if len(p) > 0:
            p = p.replace("/", os.sep).replace("\\", os.sep)
            p = p + os.sep if p[-1:] != os.sep else p
            return p

    @staticmethod
    def fix_file_path(p):
        # Only if string is nonempty
        if len(p) > 0:
            p = p.replace("/", os.sep).replace("\\", os.sep)
            return p

    # The following methods deal with config files
    def config_load(self):
        # First check if the file exists there, if not, create it
        if os.path.isfile(self.config_path + self.CONFIG_NAME):

            # Read data back from the config file and set it up in the GUI
            config = configparser.ConfigParser()
            config.read(self.config_path + self.CONFIG_NAME)

            # Before we proceed, we must ensure that all sections and options are present
            config = self.check_config(config)

            self.config_data = config

            # Set menu options
            if self.config_data['MenuOptions']['ShowLog'] == '1':
                self.actionLog.setChecked(True)
            else:
                self.actionLog.setChecked(False)

            if self.config_data['MenuOptions']['ProcessMask'] == '1':
                self.actionProcess_original_mask.setChecked(True)
            else:
                self.actionProcess_original_mask.setChecked(False)

            # Get file list, if a URL was saved
            directory = self.config_data['MenuOptions']['ImageDirectory']
            if directory != "":
                self.log('Changed working directory to ' + directory)
                self.txtImageDir.setText(directory)
                self.get_image_files()

            shpdir = self.config_data['MenuOptions']['ShapefileDirectory']
            if shpdir != "":
                self.log('Changed shapefile directory to ' + shpdir)
                #self.txtShpDir.setText(shpdir)

        else:

            # Initialize the config file
            self.config_init()

    def config_save(self):
        # If file exists (it should by now) and app initialization is finished, store new parameters
        if os.path.isfile(self.config_path + self.CONFIG_NAME) and not self.initializing:
            with open(self.config_path + self.CONFIG_NAME, 'w') as configFile:
                self.config_data.write(configFile)

    @staticmethod
    def config_defaults():

        # Dictionary
        config_defaults = {}

        # The defaults
        config_defaults['MenuOptions'] = \
            {'ShowLog': '0',
             'ProcessMask': '1',
             'ImageDirectory': '',
             'ShapefileDirectory': ''}

        return config_defaults

    def check_config(self, config):

        # Load the defaults and check whether the config has all the options
        defs = self.config_defaults()
        secs = list(defs.keys())

        # Now go item by item and add those that are missing
        for k in range(len(secs)):
            opns = list(defs[secs[k]].keys())

            # Make sure corresponding section exists
            if not config.has_section(secs[k]):
                config.add_section(secs[k])

            # And check all the options as well
            for m in range(len(opns)):
                if not config.has_option(secs[k],opns[m]):
                    config[secs[k]][opns[m]] = str(defs[secs[k]][opns[m]])

        return config

    def config_init(self):
        os.makedirs(self.config_path, exist_ok=True)  # Create the directory if needed
        config = configparser.ConfigParser()

        # Set the default configs
        the_defs = self.config_defaults()
        secs = list(the_defs.keys())
        for k in range(len(secs)):
            opns = list(the_defs[secs[k]].keys())
            config.add_section(secs[k])
            for m in range(len(opns)):
                config[secs[k]][opns[m]] = str(the_defs[secs[k]][opns[m]])

        with open(self.config_path + self.CONFIG_NAME, 'w') as configFile:
            config.write(configFile)

        self.config_data = config

    def check_paths(self):
        # Use this to check the paths
        self.txtImageDir.setText(self.fix_path(self.txtImageDir.text()))
        #self.txtShpDir.setText(self.fix_path(self.txtShpDir.text()))

    def store_paths_to_config(self):
        # Use this to store the paths to config
        self.config_data['MenuOptions']['ImageDirectory'] = self.txtImageDir.text()
        #self.config_data['MenuOptions']['ShapefileDirectory'] = self.txtShpDir.text()
        self.config_save()

    def store_menu_options_to_config(self):
        if not self.initializing:

            # Logging
            the_log = '0'
            if self.actionLog.isChecked():
                the_log = '1'
            self.config_data['MenuOptions']['ShowLog'] = the_log
            self.config_save()

    # Show different messages in status bar
    def status_bar_message(self, msgid):
        self.statusbar.showMessage(self.APP_STATUS_STATES[msgid])
        if self.app is not None:
            self.app.processEvents()

    # Locate working directory with files
    def browse_image_directory(self):
        # open hsi file
        directory = QtWidgets.QFileDialog.getOpenFileName(self, "Выбери HSi", "", "Image Files (*.mat *.h5 *.tiff *.npy)")
        if directory:
            # get type of file
            file_name_path = directory[0]             
            dot_pos_1 = len(file_name_path) - file_name_path.rfind(".")   
            file_type = file_name_path[-dot_pos_1:]
            print(file_name_path, file_type)          
              
            # get directory
            dot_pos_2 = len(file_name_path) - file_name_path.rfind("/")   #returns: Same as find, but searched right to left
            directory = file_name_path[:-dot_pos_2]                       
            self.img_name = file_name_path[-dot_pos_2:] 
            self.img_name_no_ext = file_name_path[-dot_pos_2:-dot_pos_1]
            self.img_path = file_name_path           
            print(directory, "directory")                       
            # find Key for load HSI as .mat or .h5
            if file_type == ".mat":
                hsi_data = loadmat(file_name_path)
                for key in hsi_data.keys():
                    if key[1] != "_":
                        self.key_answer = key
            elif file_type == ".h5":
                import h5py
                hsi_data = h5py.File(file_name_path, 'r')
                for key in hsi_data.keys():
                    if key[1] != "_":
                        self.key_answer = key
            # hsi            
            print(file_type)
            
            if file_type == ".mat":
                self.hsi.load_from_mat(path_to_file = file_name_path, mat_key = self.key_answer)
            elif file_type == ".tiff":
                self.hsi.load_from_tiff(path_to_file = file_name_path)
            elif file_type == ".npy":
                self.hsi.load_from_npy(path_to_file = file_name_path)
            elif file_type == ".h5":
                self.hsi.load_from_h5(path_to_file = file_name_path, h5_key = self.key_answer)

            self.loaded_hsi = self.hsi.data #
            self.wavelengths = self.hsi.wavelengths            
            print(self.wavelengths, len(self.wavelengths), "self.wavelengths, len(self.wavelengths)")     
            #print(self.wavelengths[0], self.wavelengths[-1])             
                        
            self.HSI_SLIDER_MIN = 0 #self.wavelengths[0]
            self.HSI_SLIDER_MAX = len(self.wavelengths)-1 #self.wavelengths[-1]
            self.HSI_SLIDER_DEFAULT = (self.HSI_SLIDER_MIN + self.HSI_SLIDER_MAX)/2
            self.HSI_slider.setMinimum(self.HSI_SLIDER_MIN)
            self.HSI_slider.setMaximum(self.HSI_SLIDER_MAX)
            #self.HSI_slider.setValue(self.HSI_SLIDER_DEFAULT)        
                
            # Set the path
            self.txtImageDir.setText(directory)
            self.check_paths()
            self.store_paths_to_config()

            self.log('Changed working directory to ' + directory)

            self.get_image_files()            
            self.update_annotator_view() 
            self.HSI_slider_update()
            
            # ***************
            #mask_h, mask_w, mask_k = self.hsi.data.shape
            #print(mask_h, mask_w, mask_k, "mask_h, mask_w, mask_k")
            #data = np.eye(mask_h, mask_w) # np.eye(mask_h, mask_w)
            
            ## self.hsmask = HSMask()            
            ## for i in range(mask_k):
            #self.hsmask.add_void_layer(shape=(mask_h,mask_w))
            #self.hsmask.add_completed_layer(layer=data, pos=1)            
            #print(self.hsmask.data.shape, "self.hsmask.data.shape")
                       
            # ***************
            
            #self.hsmask.
            #arr = np.zeros((100, 100, 2))
            #md = {'1':'class_1', '2':'class_2'}
            #self.hsmask = HSMask(mask=arr, label_class=md)            
            #print(self.hsmask.data, "..........self.hsmask.data")
           
            # Disable the index change event, load image, reenable it
                #self.connect_image_load_on_list_index_change(False)
            #self.load_base_mask() #  load_image
            #self.update_annotator_view() 
            #self.HSI_slider_update()
            #self.update_mask_view()             
     
            #self.connect_image_load_on_list_index_change(True)
            

    def browse_load_mask_directory(self):        
        # open hsi file
        directory = QtWidgets.QFileDialog.getOpenFileName(self, "Выбери маску", "", "Image Files (*.mat *.h5 *.tiff *.npy *.bmp *.png)")
        if directory:                       
            # get type of file
            file_name_path = directory[0]             
            dot_pos_1 = len(file_name_path) - file_name_path.rfind(".")   
            file_type = file_name_path[-dot_pos_1:]
            print(file_name_path, file_type)
                        
            # get directory
            dot_pos_2 = len(file_name_path) - file_name_path.rfind("/")   #returns: Same as find, but searched right to left
            directory = file_name_path[:-dot_pos_2]           
            
            # find Key for load mask HSI as .mat or .h5
            self.key_answer = ""
            if file_type == ".mat":
                hsi_data = loadmat(file_name_path)
                for key in hsi_data.keys():
                    if key[1] != "_":
                        self.key_answer = key
            elif file_type == ".h5":
                import h5py
                hsi_data = h5py.File(file_name_path, 'r')
                for key in hsi_data.keys():
                    if key[1] != "_":
                        self.key_answer = key                
           
            # hsmask            
            print(file_type)
            
            if file_type == ".mat":
                self.hsmask.load_from_mat(path_to_data = file_name_path, key = self.key_answer)
            elif file_type == ".tiff":
                self.hsmask.load_from_tiff(path_to_data = file_name_path)
            elif file_type == ".npy":
                self.hsmask.load_from_npy(path_to_data = file_name_path)
            elif file_type == ".h5":
                self.hsmask.load_from_h5(path_to_data = file_name_path, key = self.key_answer)
            elif file_type == ".png" or ".bmp":
                self.hsmask.load_from_image(path_to_data = file_name_path)

            self.loaded_hsmask = self.hsmask.data #
            self.label_class = self.hsmask.label_class       
            a,b,k = self.loaded_hsmask.shape      
            self.mask_layers = k
            print(self.hsmask.label_class, "self.hsmask.label_class")
            print(a,b,k, "a,b,k")
            print(self.label_class, "label_class")     
            print(type(self.loaded_hsmask), "## type(self.loaded_hsmask) ##")            
        ################################################
            self.apeend_new_lstDefectsAndColors(self.hsmask, k, is_browser_load = True, is_load_mask = True) #  self.loaded_hsmask
 

    def apeend_new_lstDefectsAndColors(self, hsmask, k, is_cluster = False, is_browser_load = False, is_load_mask = False):      # , is_ai = False
        # Create the necessary dicts
        g2rgb = {}
        rgb2g = {}        
        g2rgb_arr = {}

        max_index = self.lstDefectsAndColors.count()-1        
        print(max_index, self.cspec, "___def apeend_new_lstDefectsAndColors___: max_index, self.cspec")
        
        self.lstDefectsAndColors.clear()
        self.cspec.clear()         
        #for ind_d in range(max_index):
        #        self.cspec.pop(ind_d)
        #        self.lstDefectsAndColors.removeItem(ind_d)                
    
        for i in range(k):                                                       
            self.add_layer_current_color = self.colors_arr[i]
            #print(self.cspec, i, k, "self.cspec, i, k,")
            self.cspec.append({"NAME_LAYER_D": "New class", 
                                "COLOR_HEXRGB": self.colors_arr[i],
                                "COLOR_GSCALE_MAPPING": self.colors_gray_arr[i]})
            #print(self.cspec, i, k, "self.cspec, i, k, AFTER")
            #``  for col, row in self.cspec:            
            rgb_val = self.cspec[i]["COLOR_HEXRGB"]  
            g_val = self.cspec[i]["COLOR_GSCALE_MAPPING"]    
            name_val = self.cspec[i]["NAME_LAYER_D"]             

            #print(rgb_val,g_val, "rgb_val,g_val")
            #print("Selected: {}".format(self.colors_arr[i]), "Selected: (self.colors_arr[i])")
                        
            pix = QPixmap(50, 50)
            pix.fill(QColor(rgb_val))
            ticon = QIcon(pix)                            
            self.lstDefectsAndColors.addItem(ticon, " " + name_val +
                " | "  + rgb_val + " | "  + str(g_val))
            
            #### выгружаем слои маски и накладываем друг на друга
            #if is_ai == True:
            
            if is_cluster == True:
                layer_img = hsmask[:, :, i]  
            else:
                layer_img = hsmask.data[:, :, i] # self.loaded_hsmask[:, :, i] self.hsmask.data
           
            #from matplotlib import pyplot as PLT
            #PLT.imshow(layer_img, interpolation='nearest')
            #PLT.show()                
            
            print(type(layer_img), layer_img.shape, len(layer_img), "type(layer_img), layer_img.shape, len(layer_img)")
            
            #self.setmasktocolor(layer_img)
            #self.mask_all_layers =+ layer_img
            #print(self.mask_all_layers)
            the_color = QColor("#63" + self.add_layer_current_color.split("#")[1])
            self.current_paint = the_color
            self.annotator.brush_fill_color = the_color
            self.mask_all_colors.append(self.add_layer_current_color) #  the_color

            g2rgb_arr[g_val] = rgb_val
            rgb2g[rgb_val] = g_val  
                        
            #g2rgb[g_val] = rgb_val
            #rgb2g[rgb_val] = g_val  
            
        #self.d_rgb2gray = rgb2g
        self.d_gray2rgb_arr = g2rgb_arr
        #self.annotator.d_rgb2gray = self.d_rgb2gray
        self.annotator.d_gray2rgb_arr = self.d_gray2rgb_arr
        
        #print(self.loaded_hsmask[:, :, 1], "loaded_hsmask")          
        #print(self.mask_all_colors[1], "mask_all_colors")
        #print(self.current_image, "self.current_image")     
        
        #self.annotator.current_image = self.current_image
        # Set up dicts
        #self.d_rgb2gray = rgb2g
        #self.d_gray2rgb = g2rgb
        #self.annotator.d_rgb2gray = self.d_rgb2gray
        #self.annotator.d_gray2rgb = self.d_gray2rgb
        
        self.current_mask = layer_img
        self.current_defect = self.current_mask         
        
        '''
        if is_browser_load == True:
            new_mask = hsmask.data     
        else:  
            new_mask = hsmask 
        '''   
        # clearAndSetMaskLayers     load_SetMaskLayers
        self.annotator.load_SetMaskLayers(self.current_image, # self.current_image,
                                                hsmask, # hsmask.data[:, :, 1] self.current_mask,
                                                helper = None, # array2qimage(helper),
                                                aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                                process_gray2rgb=True, # process_gray2rgb=True,
                                                direct_mask_paint=False,
                                                color = self.mask_all_colors,
                                                data_shape = True,
                                                is_load_mask = True ) # direct_mask_paint=True))     
            
        self.check_paths()
        self.store_paths_to_config()        
        self.get_image_files()



    def QImageToCvMat(self,incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr        

    
    ################ from proj D
    def update_image(self):        
        self.mask_overlay = QPixmap(self.img_shape[0], self.img_shape[1])
        self.mask_overlay.fill(Qt.GlobalColor.transparent)

        self.crosshair_overlay = QPixmap(self.img_shape[0], self.img_shape[1])
        self.crosshair_overlay.fill(Qt.GlobalColor.transparent)

        qp = QtGui.QPainter(self.mask_overlay)
        #qp.begin(self)  # redundant with QPainter creation
        qp.end()
        qp = QtGui.QPainter(self.crosshair_overlay)

        # draw crosshairs and brush target
        #qp.setPen(QPen(QBrush(Qt.GlobalColor.yellow), 1, Qt.PenStyle.DashLine))
        #qp.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        #qp.drawLine(0, self.cursor_y, (self.img_pane_width-1),self.cursor_y)
        #qp.drawLine(self.cursor_x,0,self.cursor_x,(self.img_pane_height-1))

        # draw color brush (with correct size)
        brush_x = int(self.cursor_x - self.label_brush_size/2)
        brush_y = int(self.cursor_y - self.label_brush_size/2)
        brush_size = int(self.label_brush_size)
        qp.drawEllipse(brush_x,brush_y,brush_size,brush_size)

        qp.end()

        # draw overlay on image
        result = QPixmap(self.img_pane_width, self.img_pane_height)
        qp = QtGui.QPainter(result)
        #qp.begin(self)      # redundant with QPainter creation
        qp.drawPixmap(0,0,self.img_pixmap)
        qp.setOpacity(self.mask_opacity)
        qp.drawPixmap(0,0,self.mask_overlay)
        qp.end()

        # zoom box
        rect = self.zoom_ctrl.getCropRect()
        result = result.copy(rect).scaled(QSize(self.img_pane_width,self.img_pane_height))

        # final step: overlay crosshairs & brush reticle
        result2 = QPixmap(self.img_pane_width, self.img_pane_height)
        qp = QtGui.QPainter(result2)
        #qp.begin(self)  # redundant with QPainter creation
        qp.drawPixmap(0,0,result)
        qp.setOpacity(1.0)
        qp.drawPixmap(0,0,self.crosshair_overlay)
        qp.end()

        # blit pixmap to image pane
        self.image_pane.setPixmap(result2)

        # update image info labels
        #self.img_width = self.image.width()
        #self.img_height = self.image.height()

        self.refresh_labels()
        
    ############################################# from proj D end

    def get_image_files(self):  
        
        print("get_image_files pass..")      
        
        '''directory = self.txtImageDir.text()

        if os.path.isdir(directory):

            # Get the JPG files
            self.lstImages.clear()
            allowed_ext = [".PNG"]
            file_cnt = 0
            for file_name in os.listdir(directory):
                if any(ext in file_name.lower() for ext in allowed_ext) and file_name.count(".") < 2:
                    file_cnt += 1
                    self.lstImages.addItem(os.path.splitext(file_name)[0])

            if file_cnt > 0:
                self.dir_has_images = True
            else:
                self.dir_has_images = False

            self.log("Found " + str(file_cnt) + " images in the working directory")
        '''

    # Get black or white foreground QColor for given background color
    @staticmethod
    def get_best_fg_for_bg(color):
        r, g, b = color.getRgb()[:-1]
        fg = QColor("#ffffff")  # White by default
        if (r * 0.299 + g * 0.587 + b * 0.114) > 150:
            fg = QColor("#000000")  # Need black
        return fg

    def update_button_states(self):  # TODO: Reserved for future use
        return

    def read_defect_color_defs(self):  # Read the defect color definitions from the corresponding file
        # Read the file
        #cspec = pd.read_csv(COLOR_DEF_PATH,
        #                    delimiter=";", encoding='utf-8')
        #cspec_list = cspec.to_dict('records')
        
        df = pd.DataFrame({"NAME_LAYER_D": [], # "NAME_LAYER_D": ["New class"]
                            "COLOR_HEXRGB": [], # "COLOR_HEXRGB": ["#006f05"]
                            "COLOR_GSCALE_MAPPING": []}) # "COLOR_GSCALE_MAPPING": 1
        
        cspec_list = df.to_dict('records')

        # Store the list
        self.cspec = cspec_list       
          
        print(self.cspec, "self.cspec after")



    def add_colors_to_list(self):
        if self.cspec is not None:

            # Remove index change, if it is defined
            try:
                self.lstDefectsAndColors.disconnect()
            except:
                pass  # Do nothing, just a precaution

            # Create the necessary dicts
            g2rgb = {}
            rgb2g = {}
            tk2rgb = {}
            
            for col in self.cspec:
                rgb_val = col["COLOR_HEXRGB"].lower()
                g_val = int(col["COLOR_GSCALE_MAPPING"])

                #keys_to_insert = col["COLOR_ABBR_ET"].split(",")
                #for ks in keys_to_insert:
                #    tk2rgb[ks.strip()] = col["COLOR_HEXRGB_TK"]

                # Create the icon and populate the list
                pix = QPixmap(50, 50)
                pix.fill(QColor(rgb_val))
                ticon = QIcon(pix)
                #self.lstDefectsAndColors.addItem(ticon, " " + col["COLOR_NAME_EN"] +
                #                                 " | " + col["COLOR_NAME_ET"])
                
                self.lstDefectsAndColors.addItem(ticon, " " + col["NAME_LAYER_D"] +
                                    " | "  + col["COLOR_HEXRGB"] + " | "  + str(col["COLOR_GSCALE_MAPPING"]))

                # Fill in necessary dicts
                g2rgb[g_val] = rgb_val
                rgb2g[rgb_val] = g_val            

            # Set up dicts
            self.d_rgb2gray = rgb2g
            self.d_gray2rgb = g2rgb
            self.tk_colors = tk2rgb
            
            self.annotator.d_rgb2gray = self.d_rgb2gray
            self.annotator.d_gray2rgb = self.d_gray2rgb

            # Change the brush color
            self.lstDefectsAndColors.currentIndexChanged.connect(self.change_brush_color)

        else:
            self.log("Cannot add colors to the list, specification missing")

    def change_brush_color(self):
        cind = self.lstDefectsAndColors.currentIndex()
        
        print(cind, "change_brush_color - cind")
        if cind == -1:
            color = self.colors_arr[0]  
            the_color = QColor("#63" + color.split("#")[1])
            self.annotator.layer_mask = 0
            
        else:            
            color = self.cspec[cind]
            the_color = QColor("#63" + color["COLOR_HEXRGB"].split("#")[1]) # color["COLOR_HEXRGB"]
            self.annotator.layer_mask = cind
        self.current_paint = the_color
        self.annotator.brush_fill_color = the_color


    def delete_layerclass(self):        
        cind = self.lstDefectsAndColors.currentIndex()
        

        self.lstDefectsAndColors.removeItem(cind)

        print(self.d_gray2rgb_arr, "..........self.d_gray2rgb_arr ")
        
        self.annotator.d_gray2rgb_arr = self.d_gray2rgb_arr
        #self.annotator.d_gray2rgb_arr.pop(self.cspec[cind]["COLOR_GSCALE_MAPPING"])              
        self.cspec.pop(cind)
        print(self.cspec , "......... cspec ")        
        print(self.hsmask.data.shape, cind, "self.hsmask.shape *******************")
        
        # удаление с общей маски
        #self.add_layer_current_color
        print(self.lstDefectsAndColors.count())
        
        mask_h, mask_w, k = self.loaded_hsi.shape
        data = np.eye(mask_h, mask_w) # np.eye(mask_h, mask_w) np.zeros((mask_h,mask_w))        
         
        if self.lstDefectsAndColors.count() == 0:
            #print(self.hsmask.data.shape, " 1 ")
            #self.hsmask.add_completed_layer(layer=data, pos=cind) 
            print(self.hsmask.data.shape, " 2 ")            
            self.hsmask = HSMask() # cind+1
            print(" == 1")
        else:                         
            #print(self.hsmask.data.shape, " 1 1 ")
            #self.hsmask.add_void_layer(shape=(mask_h,mask_w), pos=cind-1)
            #self.hsmask.add_completed_layer(layer=data, pos=cind-1) 
            #print(self.hsmask.data.shape, " 2 2 ")
            self.hsmask.delete_layer(cind)
          


        print(len(self.annotator.mask_pixmap_multy), " ----------------------  ")
        self.annotator.mask_pixmap_multy.pop(cind) # self.annotator.layer_mask
        self.annotator.scene.removeItem(self.annotator._overlayHandle[cind])   
        self.annotator._overlayHandle.pop(cind) # self.annotator.layer_mask     
        print(len(self.annotator.mask_pixmap_multy), " ---------------------- mask_pixmap_multy ")
        print(len(self.annotator._overlayHandle), " ---------------------- _overlayHandle ")        
      
        self.annotator.clearAndSetMaskLayers(self.current_image, # self.current_image,
                                        self.hsmask, # hsmask.data[:, :, 1] self.current_mask,
                                        helper = None, # array2qimage(helper),
                                        aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                        process_gray2rgb=True, # process_gray2rgb=True,
                                        direct_mask_paint=False,
                                        color = self.mask_all_colors,
                                        data_shape = True,
                                        after_delete_layer = True) # data_shape = False   direct_mask_paint=True))     


    def create_hsmask_layer_for_selected_color(self, color):        
        #cind = self.lstDefectsAndColors.currentIndex()   
        cind = len(self.cspec)        
        
        if self.message_name_layer.text() == "":
            name_layer = "New class"
        else: 
            name_layer = self.message_name_layer.text()
            
        print(cind, " cind !!!!!!!")
        if cind == -1 or cind == 0:
            cind = 0

        print(color)
        self.add_layer_current_color = color
           
        self.cspec.append({"NAME_LAYER_D": name_layer, 
                            "COLOR_HEXRGB": color,
                            "COLOR_GSCALE_MAPPING": self.colors_gray_arr[cind]}) # self.colors_gray_arr[cind]
          
        #for col, row in self.cspec:            
        rgb_val = self.cspec[cind]["COLOR_HEXRGB"] #col["COLOR_HEXRGB"]
        g_val = self.cspec[cind]["COLOR_GSCALE_MAPPING"]       
        #g_val = self.cspec[cind]["NAME_LAYER_D"] # col["NAME_LAYER_D"]     
        
        # Create the icon and populate the list
        pix = QPixmap(50, 50)
        pix.fill(QColor(rgb_val))        
        ticon = QIcon(pix)
        self.lstDefectsAndColors.addItem(ticon, " " + self.cspec[cind]["NAME_LAYER_D"] +
                                         " | "  + self.add_layer_current_color + " | "  + str(self.colors_gray_arr[cind]))
        
        print(rgb_val,g_val, "Selected: {}".format(color), name_layer, "rgb_val,g_val,  .format(c), name_layer")       
        print(cind, " cind !!!!!!!")
        
        # создаем нулевую маску        
        mask_h, mask_w, k = self.loaded_hsi.shape
        
        #h, w = self.current_image.rect().height(), self.current_image.rect().width()                            
        
        data = np.eye(mask_h, mask_w) # np.eye(mask_h, mask_w) np.zeros((mask_h,mask_w))
        
        print(len(self.annotator._overlayHandle), self.hsmask.data, " zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
        if cind == 0:
            self.hsmask.add_void_layer(shape=(mask_h,mask_w))        
        else:        
            self.hsmask.add_completed_layer(layer=data, pos=cind)        
        print(len(self.annotator._overlayHandle), " zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")    
        #print(self.hsmask.data.shape, "self.hsmask.data.shape ------------------ v01")          

        ################################################
        # add_empty_mask_layer
        self.create_empty_mask_layer(self.hsmask , cind, is_browser_load = True, is_cluster = False) # arr is_cluster = True            
        self.dialog_COLOR.accept() 
        self.lstDefectsAndColors.setCurrentIndex(self.lstDefectsAndColors.count()-1)


    def create_empty_mask_layer(self, hsmask, k, is_cluster = False, is_browser_load = False):
        g2rgb = {}
        rgb2g = {}        
        g2rgb_arr = {}

        max_index = self.lstDefectsAndColors.count() - 1
        
        #print(max_index, "max_index")
        #print(self.cspec, "self.cspec")
                
        #for i in range(k):
        i = k                  
        ##############################                   
        print(self.colors_arr[i], "self.colors_arr[i] ----------------------")                                                                     
        #print(self.cspec, i, k, "self.cspec, i, k, AFTER")
       
        rgb_val = self.cspec[i]["COLOR_HEXRGB"]  
        g_val = self.cspec[i]["COLOR_GSCALE_MAPPING"]    
        name_val = self.cspec[i]["NAME_LAYER_D"]             

        print(rgb_val,g_val, "rgb_val,g_val")
        #print("Selected: {}".format(self.colors_arr[i]), "Selected: (self.colors_arr[i])")
        
        #### выгружаем слои маски и накладываем друг на друга        
        # добавляем слой в хсмаску hsmask.add_completed_layer(i, )
        if is_cluster == True:
            layer_img = self.hsmask[:, :, i]  
            print("____if is_cluster == True:")
        else:
            #hsmask.add_void_layer(i)
            layer_img = self.hsmask.data[:, :, i] # self.loaded_hsmask[:, :, i] self.hsmask.data
        
        #from matplotlib import pyplot as PLT
        #PLT.imshow(layer_img, interpolation='nearest')
        #PLT.show()                
        
        print(type(layer_img), layer_img.shape, len(layer_img), "type(layer_img), layer_img.shape, len(layer_img)")
        
        the_color = QColor("#63" + self.add_layer_current_color.split("#")[1])
        self.current_paint = the_color
        self.annotator.brush_fill_color = the_color
        
        self.mask_all_colors.append(self.add_layer_current_color) #  the_color
        print(len(self.mask_all_colors), " ................... ")
        
        g2rgb_arr[g_val] = rgb_val
        rgb2g[rgb_val] = g_val  
        ##########################################                

        #print(self.d_gray2rgb_arr, "___create_empty_mask_layer___ ---- self.d_gray2rgb_arr")
        
        if len(self.d_gray2rgb_arr) == 0:
            self.d_gray2rgb_arr = g2rgb_arr
            
        self.d_gray2rgb_arr.update(g2rgb_arr)
        
        self.annotator.d_gray2rgb_arr = self.d_gray2rgb_arr #self.annotator.d_gray2rgb_arr.update(self.d_gray2rgb_arr)
        
        #print(self.loaded_hsmask[:, :, 1], "loaded_hsmask")          
        print(self.mask_all_colors, "mask_all_colors")
        print(self.current_image, "self.current_image")     

        self.current_mask = layer_img
        self.current_defect = self.current_mask        
        
        #self.current_image=qimage2ndarray.array2qimage(self.loaded_hsi[:, :, self.HSI_slider.value()]) 

        print(self.hsmask.data.shape)
        
        # clearAndSetMaskLayers     update_MaskLayers   
        self.annotator.clearAndSetMaskLayers(self.current_image, # self.current_image,
                                                self.hsmask, # hsmask.data[:, :, 1] self.current_mask,
                                                helper = None, # array2qimage(helper),
                                                aux_helper=None, # aux_helper=(array2qimage(self.current_tk) if self.current_tk is not None else None),
                                                process_gray2rgb=True, # process_gray2rgb=True,
                                                direct_mask_paint=False,
                                                color = self.mask_all_colors,
                                                data_shape = True) # data_shape = False   direct_mask_paint=True))     
            
        self.check_paths()
        self.store_paths_to_config()        
        self.get_image_files()
        
        
    def add_layerclass_to_mask(self):        
        self.dialog_COLOR = QDialog(self , Qt.Window | Qt.WindowStaysOnTopHint) #  self , Qt.Window | Qt.WindowStaysOnTopHint
        #message_box_dialog.setModal(True)
        self.dialog_COLOR.setWindowTitle("Параметры генерации")
        self.dialog_COLOR.resize(300, 150)                    
        
        palette = PaletteGrid(self.colors_arr) # or PaletteHorizontal, or PaletteVertical   '17undertones'         
          
        message_box_label_2 = QLabel("Name your layer: ")
        self.message_name_layer = QLineEdit()                  
        palette.selected.connect(self.create_hsmask_layer_for_selected_color)        
        message_box_layout = QVBoxLayout()
        self.dialog_COLOR.setLayout(message_box_layout)
        message_box_layout.addWidget(message_box_label_2)
        message_box_layout.addWidget(self.message_name_layer)
        message_box_layout.addWidget(palette)        
        self.dialog_COLOR.show()  
        self.dialog_COLOR.exec_()                   
        
    '''
    **********
    KEY EVENTS
    **********
    '''
    def keyPressEvent(self, event):
        # Some additional shortcuts for quickly selecting defect types
        if self.annotation_mode is self.ANNOTATION_MODE_MARKING_DEFECTS:
            max_index = self.lstDefectsAndColors.count()-1
            if event.key() == Qt.Key_1:
                if max_index >= 0:
                    self.lstDefectsAndColors.setCurrentIndex(0)
            if event.key() == Qt.Key_2:
                if max_index >= 1:
                    self.lstDefectsAndColors.setCurrentIndex(1)
            if event.key() == Qt.Key_3:
                if max_index >= 2:
                    self.lstDefectsAndColors.setCurrentIndex(2)
            if event.key() == Qt.Key_4:
                if max_index >= 3:
                    self.lstDefectsAndColors.setCurrentIndex(3)
            if event.key() == Qt.Key_5:
                if max_index >= 4:
                    self.lstDefectsAndColors.setCurrentIndex(4)
            if event.key() == Qt.Key_6:
                if max_index >= 5:
                    self.lstDefectsAndColors.setCurrentIndex(5)
            if event.key() == Qt.Key_7:
                if max_index >= 6:
                    self.lstDefectsAndColors.setCurrentIndex(6)
            if event.key() == Qt.Key_8:
                if max_index >= 7:
                    self.lstDefectsAndColors.setCurrentIndex(7)
            if event.key() == Qt.Key_9:
                if max_index >= 8:
                    self.lstDefectsAndColors.setCurrentIndex(8)


def main():
    # Prepare and launch the GUI
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('annotator/res/OpenHSL_Logo_1.ico'))
    dialog = AnnotatorGUI()
    dialog.setWindowTitle(APP_TITLE + " - " + APP_VERSION) # Window title
    dialog.app = app  # Store the reference
    dialog.show()

    # Now we have to load the app configuration file
    dialog.config_load()

    # After loading the config file, we need to set up relevant UI elements
    dialog.UI_config()
    dialog.app.processEvents()

    # Now we also save the config file
    dialog.config_save()

    # And proceed with execution
    app.exec_()


# Run main loop
if __name__ == '__main__':
    # Set the exception hook
    sys.excepthook = traceback.print_exception
    main()
