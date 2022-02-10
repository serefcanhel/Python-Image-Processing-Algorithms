# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:32:43 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from forget_pass_python import Ui_MainWindow
import sqlite3

class forget(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("RESET PASSWORD")
        
        self.ui.pBut_send.clicked.connect(self.update)
        
    def update(self):
        if(int(self.ui.line_pass.text()=="") | int(self.ui.line_pass_2.text()=="") | int(self.ui.line_pass_3.text()=="" )):
            text='PLEASE FILL ALL SECTIONS !!'
            self.error(text)
        else:
            self.baglanti = sqlite3.connect("users.db")
            if(self.ui.line_pass.text() == self.ui.line_pass_2.text()):
                self.new_pass=self.ui.line_pass.text()
                self.name=self.ui.line_pass_3.text()
                self.baglanti.execute("UPDATE üyeler set parola = ? where kullanici_adi = ?",(self.new_pass,self.name))
                self.baglanti.commit()
                self.baglanti.close()
                text='YOUR PASSWORD IS CHANGED !!'
                self.error(text)
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()