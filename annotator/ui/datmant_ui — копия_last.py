# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'datmant_2wdACAj.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_DATMantMainWindow(object):
    def setupUi(self, DATMantMainWindow):
        if not DATMantMainWindow.objectName():
            DATMantMainWindow.setObjectName(u"DATMantMainWindow")
        DATMantMainWindow.resize(1288, 862)
        DATMantMainWindow.setMinimumSize(QSize(820, 620))
        self.actionLog = QAction(DATMantMainWindow)
        self.actionLog.setObjectName(u"actionLog")
        self.actionLog.setCheckable(True)
        self.actionRefresh_data_file = QAction(DATMantMainWindow)
        self.actionRefresh_data_file.setObjectName(u"actionRefresh_data_file")
        self.actionRefresh_predictor = QAction(DATMantMainWindow)
        self.actionRefresh_predictor.setObjectName(u"actionRefresh_predictor")
        self.actionSave_current_annotations = QAction(DATMantMainWindow)
        self.actionSave_current_annotations.setObjectName(u"actionSave_current_annotations")
        self.actionLoad_marked_image = QAction(DATMantMainWindow)
        self.actionLoad_marked_image.setObjectName(u"actionLoad_marked_image")
        self.actionLoad_marked_image.setCheckable(True)
        self.actionLoad_marked_image.setChecked(True)
        self.actionReload_original_mask = QAction(DATMantMainWindow)
        self.actionReload_original_mask.setObjectName(u"actionReload_original_mask")
        self.actionReload_original_mask.setEnabled(False)
        self.actionProcess_original_mask = QAction(DATMantMainWindow)
        self.actionProcess_original_mask.setObjectName(u"actionProcess_original_mask")
        self.actionProcess_original_mask.setCheckable(True)
        self.actionProcess_original_mask.setEnabled(False)
        self.actionAIMask = QAction(DATMantMainWindow)
        self.actionAIMask.setObjectName(u"actionAIMask")
        self.actionColor_definitions = QAction(DATMantMainWindow)
        self.actionColor_definitions.setObjectName(u"actionColor_definitions")
        self.actionLoad = QAction(DATMantMainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.centralwidget = QWidget(DATMantMainWindow)
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

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_4 = QLabel(self.gbAnnotWindow)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(100, 0))
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(False)
        font1.setWeight(50)
        self.label_4.setFont(font1)

        self.horizontalLayout_6.addWidget(self.label_4)

        self.lstDefectsAndColors = QComboBox(self.gbAnnotWindow)
        self.lstDefectsAndColors.setObjectName(u"lstDefectsAndColors")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lstDefectsAndColors.sizePolicy().hasHeightForWidth())
        self.lstDefectsAndColors.setSizePolicy(sizePolicy)
        self.lstDefectsAndColors.setMinimumSize(QSize(0, 30))
        self.lstDefectsAndColors.setMaximumSize(QSize(5000, 16777215))
        font2 = QFont()
        font2.setBold(False)
        font2.setWeight(50)
        self.lstDefectsAndColors.setFont(font2)

        self.horizontalLayout_6.addWidget(self.lstDefectsAndColors)

        self.makemask = QPushButton(self.gbAnnotWindow)
        self.makemask.setObjectName(u"makemask")
        self.makemask.setMinimumSize(QSize(150, 30))
        self.makemask.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_6.addWidget(self.makemask)


        self.verticalLayout_8.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.gbAnnotWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(100, 0))
        self.label_2.setFont(font1)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.txtBrushDiameter = QLineEdit(self.gbAnnotWindow)
        self.txtBrushDiameter.setObjectName(u"txtBrushDiameter")
        self.txtBrushDiameter.setMinimumSize(QSize(0, 30))
        self.txtBrushDiameter.setMaximumSize(QSize(50, 16777215))
        self.txtBrushDiameter.setFont(font2)
        self.txtBrushDiameter.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.txtBrushDiameter)

        self.sldBrushDiameter = QSlider(self.gbAnnotWindow)
        self.sldBrushDiameter.setObjectName(u"sldBrushDiameter")
        self.sldBrushDiameter.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.sldBrushDiameter)


        self.verticalLayout_8.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_7 = QLabel(self.gbAnnotWindow)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMinimumSize(QSize(100, 0))
        self.label_7.setMaximumSize(QSize(72, 16777215))
        self.label_7.setFont(font1)

        self.horizontalLayout.addWidget(self.label_7)

        self.txtImageDir = QLineEdit(self.gbAnnotWindow)
        self.txtImageDir.setObjectName(u"txtImageDir")
        self.txtImageDir.setMinimumSize(QSize(0, 30))
        self.txtImageDir.setMaximumSize(QSize(5000, 16777215))
        self.txtImageDir.setFont(font2)
        self.txtImageDir.setReadOnly(True)

        self.horizontalLayout.addWidget(self.txtImageDir)

        self.btnBrowseImageDir = QPushButton(self.gbAnnotWindow)
        self.btnBrowseImageDir.setObjectName(u"btnBrowseImageDir")
        self.btnBrowseImageDir.setMinimumSize(QSize(150, 30))
        self.btnBrowseImageDir.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout.addWidget(self.btnBrowseImageDir)


        self.verticalLayout_8.addLayout(self.horizontalLayout)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_5 = QLabel(self.gbAnnotWindow)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(100, 0))
        self.label_5.setFont(font1)

        self.horizontalLayout_7.addWidget(self.label_5)

        self.txt_HSI_slider = QLineEdit(self.gbAnnotWindow)
        self.txt_HSI_slider.setObjectName(u"txt_HSI_slider")
        self.txt_HSI_slider.setMinimumSize(QSize(0, 30))
        self.txt_HSI_slider.setMaximumSize(QSize(50, 16777215))
        self.txt_HSI_slider.setReadOnly(True)

        self.horizontalLayout_7.addWidget(self.txt_HSI_slider)

        self.txt_HSI_slider_leyer = QLineEdit(self.gbAnnotWindow)
        self.txt_HSI_slider_leyer.setObjectName(u"txt_HSI_slider_leyer")
        self.txt_HSI_slider_leyer.setMinimumSize(QSize(0, 30))
        self.txt_HSI_slider_leyer.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_7.addWidget(self.txt_HSI_slider_leyer)

        self.HSI_slider = QSlider(self.gbAnnotWindow)
        self.HSI_slider.setObjectName(u"HSI_slider")
        self.HSI_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_7.addWidget(self.HSI_slider)


        self.verticalLayout_8.addLayout(self.horizontalLayout_7)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.lstImages = QComboBox(self.gbAnnotWindow)
        self.lstImages.setObjectName(u"lstImages")
        self.lstImages.setFont(font2)

        self.gridLayout.addWidget(self.lstImages, 2, 2, 1, 3)

        self.txtImageHasDefectMask = QLineEdit(self.gbAnnotWindow)
        self.txtImageHasDefectMask.setObjectName(u"txtImageHasDefectMask")
        self.txtImageHasDefectMask.setFont(font2)
        self.txtImageHasDefectMask.setReadOnly(True)

        self.gridLayout.addWidget(self.txtImageHasDefectMask, 3, 4, 1, 1)

        self.label_15 = QLabel(self.gbAnnotWindow)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setMinimumSize(QSize(100, 0))
        self.label_15.setFont(font1)

        self.gridLayout.addWidget(self.label_15, 2, 0, 1, 1)

        self.label = QLabel(self.gbAnnotWindow)
        self.label.setObjectName(u"label")
        self.label.setFont(font1)
        self.label.setTextFormat(Qt.PlainText)

        self.gridLayout.addWidget(self.label, 3, 3, 1, 1)

        self.txtImageStatus = QLineEdit(self.gbAnnotWindow)
        self.txtImageStatus.setObjectName(u"txtImageStatus")
        self.txtImageStatus.setFont(font2)
        self.txtImageStatus.setReadOnly(True)

        self.gridLayout.addWidget(self.txtImageStatus, 3, 2, 1, 1)

        self.label_14 = QLabel(self.gbAnnotWindow)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMinimumSize(QSize(100, 0))
        self.label_14.setFont(font1)

        self.gridLayout.addWidget(self.label_14, 3, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)


        self.verticalLayout_8.addLayout(self.verticalLayout)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.gbAnnotWindow)
        self.label_3.setObjectName(u"label_3")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy1)
        self.label_3.setMinimumSize(QSize(100, 0))
        self.label_3.setFont(font1)

        self.horizontalLayout_5.addWidget(self.label_3)

        self.txtShpDir = QLineEdit(self.gbAnnotWindow)
        self.txtShpDir.setObjectName(u"txtShpDir")
        self.txtShpDir.setMinimumSize(QSize(0, 30))
        self.txtShpDir.setMaximumSize(QSize(5000, 16777215))
        self.txtShpDir.setFont(font2)
        self.txtShpDir.setReadOnly(True)

        self.horizontalLayout_5.addWidget(self.txtShpDir)

        self.btnBrowseShp = QPushButton(self.gbAnnotWindow)
        self.btnBrowseShp.setObjectName(u"btnBrowseShp")
        self.btnBrowseShp.setMinimumSize(QSize(150, 30))
        self.btnBrowseShp.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_5.addWidget(self.btnBrowseShp)


        self.verticalLayout_8.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.btnPrev = QPushButton(self.gbAnnotWindow)
        self.btnPrev.setObjectName(u"btnPrev")

        self.horizontalLayout_4.addWidget(self.btnPrev)

        self.btnNext = QPushButton(self.gbAnnotWindow)
        self.btnNext.setObjectName(u"btnNext")

        self.horizontalLayout_4.addWidget(self.btnNext)

        self.btnClear = QPushButton(self.gbAnnotWindow)
        self.btnClear.setObjectName(u"btnClear")
        self.btnClear.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.btnClear)

        self.btnMode = QPushButton(self.gbAnnotWindow)
        self.btnMode.setObjectName(u"btnMode")

        self.horizontalLayout_4.addWidget(self.btnMode)


        self.verticalLayout_8.addLayout(self.horizontalLayout_4)


        self.verticalLayout_2.addWidget(self.gbAnnotWindow)

        self.gbApplicationLog = QGroupBox(self.centralwidget)
        self.gbApplicationLog.setObjectName(u"gbApplicationLog")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.gbApplicationLog.sizePolicy().hasHeightForWidth())
        self.gbApplicationLog.setSizePolicy(sizePolicy2)
        self.gbApplicationLog.setMaximumSize(QSize(16777215, 150))
        self.gbApplicationLog.setFont(font)
        self.verticalLayout_3 = QVBoxLayout(self.gbApplicationLog)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.txtConsole = QTextEdit(self.gbApplicationLog)
        self.txtConsole.setObjectName(u"txtConsole")
        self.txtConsole.setFont(font2)

        self.verticalLayout_3.addWidget(self.txtConsole)


        self.verticalLayout_2.addWidget(self.gbApplicationLog)

        DATMantMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(DATMantMainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1288, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menuEdit")
        DATMantMainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(DATMantMainWindow)
        self.statusbar.setObjectName(u"statusbar")
        DATMantMainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menuFile.addAction(self.actionSave_current_annotations)
        self.menuFile.addAction(self.actionReload_original_mask)
        self.menuFile.addAction(self.actionLoad)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionColor_definitions)
        self.menuView.addAction(self.actionLog)
        self.menuEdit.addAction(self.actionProcess_original_mask)
        self.menuEdit.addAction(self.actionAIMask)

        self.retranslateUi(DATMantMainWindow)

        QMetaObject.connectSlotsByName(DATMantMainWindow)
    # setupUi

    def retranslateUi(self, DATMantMainWindow):
        DATMantMainWindow.setWindowTitle(QCoreApplication.translate("DATMantMainWindow", u"Annotator for HSI", None))
        self.actionLog.setText(QCoreApplication.translate("DATMantMainWindow", u"Log", None))
        self.actionRefresh_data_file.setText(QCoreApplication.translate("DATMantMainWindow", u"Refresh data file", None))
        self.actionRefresh_predictor.setText(QCoreApplication.translate("DATMantMainWindow", u"Refresh predictor", None))
        self.actionSave_current_annotations.setText(QCoreApplication.translate("DATMantMainWindow", u"Save current annotations", None))
        self.actionLoad_marked_image.setText(QCoreApplication.translate("DATMantMainWindow", u"Load marked image", None))
        self.actionReload_original_mask.setText(QCoreApplication.translate("DATMantMainWindow", u"Reload original mask", None))
        self.actionProcess_original_mask.setText(QCoreApplication.translate("DATMantMainWindow", u"Process original mask", None))
        self.actionAIMask.setText(QCoreApplication.translate("DATMantMainWindow", u"Reload AUTO defect mask", None))
        self.actionColor_definitions.setText(QCoreApplication.translate("DATMantMainWindow", u"Color specifications", None))
        self.actionLoad.setText(QCoreApplication.translate("DATMantMainWindow", u"Load annotations", None))
        self.gbAnnotWindow.setTitle(QCoreApplication.translate("DATMantMainWindow", u"Annotation window", None))
        self.label_4.setText(QCoreApplication.translate("DATMantMainWindow", u"Annotated layer:", None))
        self.makemask.setText(QCoreApplication.translate("DATMantMainWindow", u"Smart Mask", None))
        self.label_2.setText(QCoreApplication.translate("DATMantMainWindow", u"Brush diameter:", None))
        self.label_7.setText(QCoreApplication.translate("DATMantMainWindow", u"Folder HSI:", None))
        self.btnBrowseImageDir.setText(QCoreApplication.translate("DATMantMainWindow", u"Open HSI", None))
        self.label_5.setText(QCoreApplication.translate("DATMantMainWindow", u"Wavelengths / Layer HSI", None))
        self.label_15.setText(QCoreApplication.translate("DATMantMainWindow", u"Current layer:", None))
        self.label.setText(QCoreApplication.translate("DATMantMainWindow", u"Has updated mask?", None))
        self.label_14.setText(QCoreApplication.translate("DATMantMainWindow", u"Image status:", None))
        self.label_3.setText(QCoreApplication.translate("DATMantMainWindow", u"Mask .shp folder:", None))
        self.btnBrowseShp.setText(QCoreApplication.translate("DATMantMainWindow", u"Browse...", None))
        self.btnPrev.setText(QCoreApplication.translate("DATMantMainWindow", u"[P] Previous image", None))
#if QT_CONFIG(shortcut)
        self.btnPrev.setShortcut(QCoreApplication.translate("DATMantMainWindow", u"P", None))
#endif // QT_CONFIG(shortcut)
        self.btnNext.setText(QCoreApplication.translate("DATMantMainWindow", u"[N] Next image", None))
#if QT_CONFIG(shortcut)
        self.btnNext.setShortcut(QCoreApplication.translate("DATMantMainWindow", u"N", None))
#endif // QT_CONFIG(shortcut)
        self.btnClear.setText(QCoreApplication.translate("DATMantMainWindow", u"Clear current annotations", None))
        self.btnMode.setText(QCoreApplication.translate("DATMantMainWindow", u"Mode [Marking defects]", None))
#if QT_CONFIG(shortcut)
        self.btnMode.setShortcut(QCoreApplication.translate("DATMantMainWindow", u"M", None))
#endif // QT_CONFIG(shortcut)
        self.gbApplicationLog.setTitle(QCoreApplication.translate("DATMantMainWindow", u"Application log", None))
        self.menuFile.setTitle(QCoreApplication.translate("DATMantMainWindow", u"File", None))
        self.menuView.setTitle(QCoreApplication.translate("DATMantMainWindow", u"View", None))
        self.menuEdit.setTitle(QCoreApplication.translate("DATMantMainWindow", u"Edit", None))
    # retranslateUi

