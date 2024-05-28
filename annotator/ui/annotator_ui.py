# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'annotator_5ATiXWE.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_AnnotatorMainWindow(object):
    def setupUi(self, AnnotatorMainWindow):
        if not AnnotatorMainWindow.objectName():
            AnnotatorMainWindow.setObjectName(u"AnnotatorMainWindow")
        AnnotatorMainWindow.resize(1288, 862)
        AnnotatorMainWindow.setMinimumSize(QSize(820, 620))
        self.actionLog = QAction(AnnotatorMainWindow)
        self.actionLog.setObjectName(u"actionLog")
        self.actionLog.setCheckable(True)
        self.actionRefresh_data_file = QAction(AnnotatorMainWindow)
        self.actionRefresh_data_file.setObjectName(u"actionRefresh_data_file")
        self.actionRefresh_predictor = QAction(AnnotatorMainWindow)
        self.actionRefresh_predictor.setObjectName(u"actionRefresh_predictor")
        self.actionSave_current_annotations = QAction(AnnotatorMainWindow)
        self.actionSave_current_annotations.setObjectName(u"actionSave_current_annotations")
        self.actionLoad_marked_image = QAction(AnnotatorMainWindow)
        self.actionLoad_marked_image.setObjectName(u"actionLoad_marked_image")
        self.actionLoad_marked_image.setCheckable(True)
        self.actionLoad_marked_image.setChecked(True)
        self.actionReload_original_mask = QAction(AnnotatorMainWindow)
        self.actionReload_original_mask.setObjectName(u"actionReload_original_mask")
        self.actionReload_original_mask.setEnabled(False)
        self.actionProcess_original_mask = QAction(AnnotatorMainWindow)
        self.actionProcess_original_mask.setObjectName(u"actionProcess_original_mask")
        self.actionProcess_original_mask.setCheckable(True)
        self.actionProcess_original_mask.setEnabled(False)
        self.actionAIMask = QAction(AnnotatorMainWindow)
        self.actionAIMask.setObjectName(u"actionAIMask")
        self.actionColor_definitions = QAction(AnnotatorMainWindow)
        self.actionColor_definitions.setObjectName(u"actionColor_definitions")
        self.actionLoad = QAction(AnnotatorMainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.centralwidget = QWidget(AnnotatorMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gbAnnotWindow = QGroupBox(self.centralwidget)
        self.gbAnnotWindow.setObjectName(u"gbAnnotWindow")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbAnnotWindow.setFont(font)
        self.verticalLayout_8 = QVBoxLayout(self.gbAnnotWindow)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.figThinFigure = QVBoxLayout()
        self.figThinFigure.setObjectName(u"figThinFigure")

        self.horizontalLayout_3.addLayout(self.figThinFigure)


        self.verticalLayout_8.addLayout(self.horizontalLayout_3)

        self.line = QFrame(self.gbAnnotWindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_8.addWidget(self.line)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, -1, 0, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setSizeConstraint(QLayout.SetMinimumSize)
        self.horizontalLayout_5.setContentsMargins(-1, -1, 0, -1)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.label_5 = QLabel(self.gbAnnotWindow)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(100, 0))
        self.label_5.setMaximumSize(QSize(200, 16777215))
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(False)
        font1.setWeight(50)
        self.label_5.setFont(font1)

        self.horizontalLayout_5.addWidget(self.label_5)

        self.txt_HSI_slider = QLineEdit(self.gbAnnotWindow)
        self.txt_HSI_slider.setObjectName(u"txt_HSI_slider")
        self.txt_HSI_slider.setMinimumSize(QSize(0, 30))
        self.txt_HSI_slider.setMaximumSize(QSize(50, 16777215))
        self.txt_HSI_slider.setReadOnly(True)

        self.horizontalLayout_5.addWidget(self.txt_HSI_slider)

        self.label = QLabel(self.gbAnnotWindow)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(5, 0))

        self.horizontalLayout_5.addWidget(self.label)

        self.txt_HSI_slider_leyer = QLineEdit(self.gbAnnotWindow)
        self.txt_HSI_slider_leyer.setObjectName(u"txt_HSI_slider_leyer")
        self.txt_HSI_slider_leyer.setMinimumSize(QSize(0, 30))
        self.txt_HSI_slider_leyer.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_5.addWidget(self.txt_HSI_slider_leyer)

        self.label_3 = QLabel(self.gbAnnotWindow)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(5, 0))

        self.horizontalLayout_5.addWidget(self.label_3)

        self.HSI_slider = QSlider(self.gbAnnotWindow)
        self.HSI_slider.setObjectName(u"HSI_slider")
        self.HSI_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_5.addWidget(self.HSI_slider)


        self.verticalLayout_4.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setSizeConstraint(QLayout.SetMinimumSize)
        self.horizontalLayout_8.setContentsMargins(-1, -1, 0, -1)
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_3)

        self.label_2 = QLabel(self.gbAnnotWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(100, 0))
        self.label_2.setMaximumSize(QSize(200, 16777215))
        self.label_2.setFont(font1)

        self.horizontalLayout_8.addWidget(self.label_2)

        self.txtBrushDiameter = QLineEdit(self.gbAnnotWindow)
        self.txtBrushDiameter.setObjectName(u"txtBrushDiameter")
        self.txtBrushDiameter.setMinimumSize(QSize(0, 30))
        self.txtBrushDiameter.setMaximumSize(QSize(50, 16777215))
        font2 = QFont()
        font2.setBold(False)
        font2.setWeight(50)
        self.txtBrushDiameter.setFont(font2)
        self.txtBrushDiameter.setReadOnly(True)

        self.horizontalLayout_8.addWidget(self.txtBrushDiameter)

        self.label_6 = QLabel(self.gbAnnotWindow)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMinimumSize(QSize(5, 0))

        self.horizontalLayout_8.addWidget(self.label_6)

        self.sldBrushDiameter = QSlider(self.gbAnnotWindow)
        self.sldBrushDiameter.setObjectName(u"sldBrushDiameter")
        self.sldBrushDiameter.setOrientation(Qt.Horizontal)

        self.horizontalLayout_8.addWidget(self.sldBrushDiameter)


        self.verticalLayout_4.addLayout(self.horizontalLayout_8)

        self.label_13 = QLabel(self.gbAnnotWindow)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_4.addWidget(self.label_13)


        self.horizontalLayout_7.addLayout(self.verticalLayout_4)


        self.verticalLayout_8.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(-1, 0, -1, -1)
        self.btnBrowseImageDir = QPushButton(self.gbAnnotWindow)
        self.btnBrowseImageDir.setObjectName(u"btnBrowseImageDir")
        self.btnBrowseImageDir.setMinimumSize(QSize(150, 30))
        self.btnBrowseImageDir.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_13.addWidget(self.btnBrowseImageDir)

        self.label_7 = QLabel(self.gbAnnotWindow)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMinimumSize(QSize(0, 0))
        self.label_7.setMaximumSize(QSize(100, 16777215))
        self.label_7.setFont(font1)
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_13.addWidget(self.label_7)

        self.txtImageDir = QLineEdit(self.gbAnnotWindow)
        self.txtImageDir.setObjectName(u"txtImageDir")
        self.txtImageDir.setMinimumSize(QSize(0, 30))
        self.txtImageDir.setMaximumSize(QSize(360, 16777215))
        self.txtImageDir.setFont(font2)
        self.txtImageDir.setReadOnly(True)

        self.horizontalLayout_13.addWidget(self.txtImageDir)


        self.verticalLayout_11.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(-1, 10, -1, -1)
        self.actionLoadmask = QPushButton(self.gbAnnotWindow)
        self.actionLoadmask.setObjectName(u"actionLoadmask")
        self.actionLoadmask.setMinimumSize(QSize(150, 30))
        self.actionLoadmask.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_14.addWidget(self.actionLoadmask)

        self.label_9 = QLabel(self.gbAnnotWindow)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMaximumSize(QSize(100, 16777215))
        self.label_9.setFont(font1)
        self.label_9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_14.addWidget(self.label_9)

        self.lineEdit = QLineEdit(self.gbAnnotWindow)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setMinimumSize(QSize(0, 30))
        self.lineEdit.setMaximumSize(QSize(360, 16777215))
        self.lineEdit.setFont(font2)
        self.lineEdit.setReadOnly(True)

        self.horizontalLayout_14.addWidget(self.lineEdit)


        self.verticalLayout_11.addLayout(self.horizontalLayout_14)

        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(-1, -1, 10, -1)
        self.label_12 = QLabel(self.gbAnnotWindow)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_15.addWidget(self.label_12)

        self.label_11 = QLabel(self.gbAnnotWindow)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_15.addWidget(self.label_11)


        self.verticalLayout_13.addLayout(self.horizontalLayout_15)


        self.verticalLayout_11.addLayout(self.verticalLayout_13)


        self.horizontalLayout.addLayout(self.verticalLayout_11)


        self.horizontalLayout_6.addLayout(self.horizontalLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(20, -1, 10, -1)
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_10 = QLabel(self.gbAnnotWindow)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMaximumSize(QSize(16777215, 200))
        self.label_10.setFont(font1)
        self.label_10.setAlignment(Qt.AlignBottom|Qt.AlignLeading|Qt.AlignLeft)

        self.verticalLayout_6.addWidget(self.label_10)


        self.verticalLayout_5.addLayout(self.verticalLayout_6)

        self.lstDefectsAndColors = QComboBox(self.gbAnnotWindow)
        self.lstDefectsAndColors.setObjectName(u"lstDefectsAndColors")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lstDefectsAndColors.sizePolicy().hasHeightForWidth())
        self.lstDefectsAndColors.setSizePolicy(sizePolicy)
        self.lstDefectsAndColors.setMinimumSize(QSize(0, 30))
        self.lstDefectsAndColors.setMaximumSize(QSize(5000, 16777215))
        self.lstDefectsAndColors.setFont(font2)

        self.verticalLayout_5.addWidget(self.lstDefectsAndColors)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setSpacing(3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 5, -1, -1)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(3)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(-1, 0, -1, -1)
        self.add_layer = QPushButton(self.gbAnnotWindow)
        self.add_layer.setObjectName(u"add_layer")
        self.add_layer.setMinimumSize(QSize(0, 30))
        self.add_layer.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_9.addWidget(self.add_layer)

        self.delete_layer = QPushButton(self.gbAnnotWindow)
        self.delete_layer.setObjectName(u"delete_layer")
        self.delete_layer.setMinimumSize(QSize(0, 30))
        self.delete_layer.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_9.addWidget(self.delete_layer)

        self.btnMode = QPushButton(self.gbAnnotWindow)
        self.btnMode.setObjectName(u"btnMode")
        self.btnMode.setMinimumSize(QSize(0, 30))
        self.btnMode.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_9.addWidget(self.btnMode)


        self.verticalLayout_7.addLayout(self.horizontalLayout_9)


        self.verticalLayout_5.addLayout(self.verticalLayout_7)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.label_4 = QLabel(self.gbAnnotWindow)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_16.addWidget(self.label_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_16)


        self.horizontalLayout_4.addLayout(self.verticalLayout_5)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, -1, 10, -1)
        self.makemask = QPushButton(self.gbAnnotWindow)
        self.makemask.setObjectName(u"makemask")
        self.makemask.setMinimumSize(QSize(150, 30))
        self.makemask.setMaximumSize(QSize(300, 16777215))

        self.verticalLayout.addWidget(self.makemask)

        self.btnClear = QPushButton(self.gbAnnotWindow)
        self.btnClear.setObjectName(u"btnClear")
        self.btnClear.setEnabled(True)
        self.btnClear.setMinimumSize(QSize(0, 30))
        self.btnClear.setMaximumSize(QSize(300, 16777215))

        self.verticalLayout.addWidget(self.btnClear)

        self.InvertingMask = QPushButton(self.gbAnnotWindow)
        self.InvertingMask.setObjectName(u"InvertingMask")
        self.InvertingMask.setMinimumSize(QSize(0, 30))

        self.verticalLayout.addWidget(self.InvertingMask)

        self.pushButton_2 = QPushButton(self.gbAnnotWindow)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setMinimumSize(QSize(150, 30))
        self.pushButton_2.setMaximumSize(QSize(200, 16777215))

        self.verticalLayout.addWidget(self.pushButton_2)


        self.horizontalLayout_4.addLayout(self.verticalLayout)


        self.horizontalLayout_6.addLayout(self.horizontalLayout_4)


        self.verticalLayout_8.addLayout(self.horizontalLayout_6)


        self.verticalLayout_2.addWidget(self.gbAnnotWindow)

        self.gbApplicationLog = QGroupBox(self.centralwidget)
        self.gbApplicationLog.setObjectName(u"gbApplicationLog")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.gbApplicationLog.sizePolicy().hasHeightForWidth())
        self.gbApplicationLog.setSizePolicy(sizePolicy1)
        self.gbApplicationLog.setMaximumSize(QSize(16777215, 150))
        self.gbApplicationLog.setFont(font)
        self.verticalLayout_3 = QVBoxLayout(self.gbApplicationLog)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.txtConsole = QTextEdit(self.gbApplicationLog)
        self.txtConsole.setObjectName(u"txtConsole")
        self.txtConsole.setFont(font2)

        self.verticalLayout_3.addWidget(self.txtConsole)


        self.verticalLayout_2.addWidget(self.gbApplicationLog)

        AnnotatorMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(AnnotatorMainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1288, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menuEdit")
        AnnotatorMainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(AnnotatorMainWindow)
        self.statusbar.setObjectName(u"statusbar")
        AnnotatorMainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionLog)

        self.retranslateUi(AnnotatorMainWindow)

        QMetaObject.connectSlotsByName(AnnotatorMainWindow)
    # setupUi

    def retranslateUi(self, AnnotatorMainWindow):
        AnnotatorMainWindow.setWindowTitle(QCoreApplication.translate("AnnotatorMainWindow", u"Annotator for HSI", None))
        self.actionLog.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Log", None))
        self.actionRefresh_data_file.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Refresh data file", None))
        self.actionRefresh_predictor.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Refresh predictor", None))
        self.actionSave_current_annotations.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Save current annotations", None))
        self.actionLoad_marked_image.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Load marked image", None))
        self.actionReload_original_mask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Reload original mask", None))
        self.actionProcess_original_mask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Process original mask", None))
        self.actionAIMask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Reload AUTO defect mask", None))
        self.actionColor_definitions.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Color specifications", None))
        self.actionLoad.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Load annotations", None))
        self.gbAnnotWindow.setTitle(QCoreApplication.translate("AnnotatorMainWindow", u"Annotation", None))
        self.label_5.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Wavelengths / Layer HSI:  ", None))
        self.label.setText("")
        self.label_3.setText("")
        self.label_2.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Brush diameter:", None))
        self.label_6.setText("")
        self.label_13.setText("")
        self.btnBrowseImageDir.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Open HSI", None))
        self.label_7.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Folder HSI:", None))
        self.actionLoadmask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Load Mask", None))
        self.label_9.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Folder MASK:", None))
        self.label_12.setText("")
        self.label_11.setText("")
        self.label_10.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Annotated layer:", None))
        self.add_layer.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Add layer", None))
        self.delete_layer.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Delete layer", None))
        self.btnMode.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Hide layers", None))
#if QT_CONFIG(shortcut)
        self.btnMode.setShortcut(QCoreApplication.translate("AnnotatorMainWindow", u"M", None))
#endif // QT_CONFIG(shortcut)
        self.label_4.setText("")
        self.makemask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Smart Mask", None))
        self.btnClear.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Clear Mask", None))
        self.InvertingMask.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Inverting Mask", None))
        self.pushButton_2.setText(QCoreApplication.translate("AnnotatorMainWindow", u"Save Mask", None))
        self.gbApplicationLog.setTitle(QCoreApplication.translate("AnnotatorMainWindow", u"Application log", None))
        self.menuFile.setTitle(QCoreApplication.translate("AnnotatorMainWindow", u"File", None))
        self.menuView.setTitle(QCoreApplication.translate("AnnotatorMainWindow", u"View", None))
        self.menuEdit.setTitle(QCoreApplication.translate("AnnotatorMainWindow", u"Edit", None))
    # retranslateUi

