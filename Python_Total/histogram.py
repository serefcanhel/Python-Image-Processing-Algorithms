# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:10:50 2021

@author: SEREF
"""
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class histogram(QWidget):

    def _init_(self, parent=None):
        QWidget._init_(self, parent)
        vertical_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.figure.patch.set_facecolor("None")
        self.canvas = FigureCanvas(self.figure)
        vertical_layout.addWidget(self.canvas)
        self.canvas.setStyleSheet("background-color:transparent;")
        
        self.canvas.axes = self.canvas.figure.add_subplot(111,position=[0, 0, 1, 1])
        self.setLayout(vertical_layout)