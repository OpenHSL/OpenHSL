# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'color_specs_2EgGhEL.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_ColorSpecsUI(object):
    def setupUi(self, ColorSpecsUI):
        if not ColorSpecsUI.objectName():
            ColorSpecsUI.setObjectName(u"ColorSpecsUI")
        ColorSpecsUI.resize(520, 386)
        self.centralwidget = QWidget(ColorSpecsUI)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.tabColorSpecs = QTableWidget(self.centralwidget)
        self.tabColorSpecs.setObjectName(u"tabColorSpecs")

        self.verticalLayout.addWidget(self.tabColorSpecs)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.add_new_layer_pb = QPushButton(self.centralwidget)
        self.add_new_layer_pb.setObjectName(u"add_new_layer_pb")

        self.horizontalLayout_2.addWidget(self.add_new_layer_pb)

        self.change_layer_data = QPushButton(self.centralwidget)
        self.change_layer_data.setObjectName(u"change_layer_data")

        self.horizontalLayout_2.addWidget(self.change_layer_data)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        ColorSpecsUI.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(ColorSpecsUI)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 520, 22))
        ColorSpecsUI.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(ColorSpecsUI)
        self.statusbar.setObjectName(u"statusbar")
        ColorSpecsUI.setStatusBar(self.statusbar)

        self.retranslateUi(ColorSpecsUI)

        QMetaObject.connectSlotsByName(ColorSpecsUI)
    # setupUi

    def retranslateUi(self, ColorSpecsUI):
        ColorSpecsUI.setWindowTitle(QCoreApplication.translate("ColorSpecsUI", u"Color specifications", None))
        self.add_new_layer_pb.setText(QCoreApplication.translate("ColorSpecsUI", u"Add New Layer", None))
        self.change_layer_data.setText(QCoreApplication.translate("ColorSpecsUI", u"Apply", None))
    # retranslateUi

