# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui\color_specs.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ColorSpecsUI(object):
    def setupUi(self, ColorSpecsUI):
        ColorSpecsUI.setObjectName("ColorSpecsUI")
        ColorSpecsUI.resize(520, 380)
        self.centralwidget = QtWidgets.QWidget(ColorSpecsUI)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabColorSpecs = QtWidgets.QTableWidget(self.centralwidget)
        self.tabColorSpecs.setObjectName("tabColorSpecs")
        self.tabColorSpecs.setColumnCount(0)
        self.tabColorSpecs.setRowCount(0)
        self.verticalLayout.addWidget(self.tabColorSpecs)
        ColorSpecsUI.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ColorSpecsUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 520, 21))
        self.menubar.setObjectName("menubar")
        ColorSpecsUI.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ColorSpecsUI)
        self.statusbar.setObjectName("statusbar")
        ColorSpecsUI.setStatusBar(self.statusbar)

        self.retranslateUi(ColorSpecsUI)
        QtCore.QMetaObject.connectSlotsByName(ColorSpecsUI)

    def retranslateUi(self, ColorSpecsUI):
        _translate = QtCore.QCoreApplication.translate
        ColorSpecsUI.setWindowTitle(_translate("ColorSpecsUI", "Color specifications"))

