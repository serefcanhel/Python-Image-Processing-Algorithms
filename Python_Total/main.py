# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:36:34 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from login import Main


# MAIN APPLICATION
app=QApplication([])
window=Main()
window.show()

app.exec_()