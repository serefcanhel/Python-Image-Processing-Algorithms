# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/SEREF/Desktop/dersnotlari/4.year/FallTerm/ImageProcessing/HMW/2/PYTHON/Login.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(566, 547)
        MainWindow.setStyleSheet("\n"
" background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0.965909, stop:0 rgba(78, 237, 163, 255), stop:0.248588 rgba(0, 107, 128, 255), stop:0.723164 rgba(48, 150, 223, 255), stop:1 rgba(3, 131, 103, 255));")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 541, 211))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(8)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 230, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(80, 290, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setMouseTracking(True)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_2.setLineWidth(1)
        self.label_2.setScaledContents(False)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.line_username = QtWidgets.QLineEdit(self.centralwidget)
        self.line_username.setGeometry(QtCore.QRect(220, 230, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Leelawadee UI")
        self.line_username.setFont(font)
        self.line_username.setObjectName("line_username")
        self.line_pass = QtWidgets.QLineEdit(self.centralwidget)
        self.line_pass.setGeometry(QtCore.QRect(220, 290, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Leelawadee UI")
        self.line_pass.setFont(font)
        self.line_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.line_pass.setObjectName("line_pass")
        self.pBut_change = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_change.setGeometry(QtCore.QRect(420, 290, 31, 41))
        self.pBut_change.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/indir.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pBut_change.setIcon(icon)
        self.pBut_change.setIconSize(QtCore.QSize(40, 40))
        self.pBut_change.setObjectName("pBut_change")
        self.pBut_login = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_login.setGeometry(QtCore.QRect(70, 360, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        self.pBut_login.setFont(font)
        self.pBut_login.setObjectName("pBut_login")
        self.pBut_forget = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_forget.setGeometry(QtCore.QRect(240, 360, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        self.pBut_forget.setFont(font)
        self.pBut_forget.setObjectName("pBut_forget")
        self.pBut_continue = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_continue.setGeometry(QtCore.QRect(250, 420, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        self.pBut_continue.setFont(font)
        self.pBut_continue.setObjectName("pBut_continue")
        self.pBut_register = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_register.setGeometry(QtCore.QRect(30, 420, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(12)
        self.pBut_register.setFont(font)
        self.pBut_register.setObjectName("pBut_register")
        self.pBut_school = QtWidgets.QPushButton(self.centralwidget)
        self.pBut_school.setGeometry(QtCore.QRect(10, 460, 51, 41))
        self.pBut_school.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/indir.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pBut_school.setIcon(icon1)
        self.pBut_school.setIconSize(QtCore.QSize(50, 50))
        self.pBut_school.setObjectName("pBut_school")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 566, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; color:#010404;\">EEE-410 IMAGE PROCESSING</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color:#040404;\">Username</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#060606;\">Password</span></p></body></html>"))
        self.pBut_login.setText(_translate("MainWindow", "LOGIN"))
        self.pBut_forget.setText(_translate("MainWindow", "FORGET PASSWORD"))
        self.pBut_continue.setText(_translate("MainWindow", "CONTINUE WITHOUT LOGIN"))
        self.pBut_register.setText(_translate("MainWindow", "CREATE AN ACCOUNT"))

import icons_rc
