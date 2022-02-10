# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:37:15 2021

@author: SEREF
"""
from Login_python import Ui_Form
from PyQt5.QtWidgets import*
import sqlite3
from PyQt5.QtGui import QIcon
import webbrowser
from process import Process
from register import register
from forgetPass import forget

class Main(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Main Menu")
        
        self.process=Process()
        self.register=register()
        self.forget=forget()
        
        self.ui.pBut_continue.clicked.connect(self.OpenConversion)
        self.ui.pBut_login.clicked.connect(self.login)
        self.ui.pBut_change.clicked.connect(self.change)
        self.ui.pBut_school.clicked.connect(self.web)
        self.ui.pBut_register.clicked.connect(self.RegisterPage)
        self.ui.pBut_forget.clicked.connect(self.ResetPass)
        self.PassCheck=1
        
    def OpenConversion(self):
        self.process.show()
        
    def RegisterPage(self):
        self.register.show()
        
    def ResetPass(self):
        self.forget.show()
        
    def web(self):
        webbrowser.open("https://www.ikcu.edu.tr/")
        
    def change(self):
        if self.PassCheck ==1:
            self.ui.line_pass.setEchoMode(QLineEdit.EchoMode.Normal)
            self.PassCheck=0
        else:
            self.ui.line_pass.setEchoMode(QLineEdit.EchoMode.Password)
            self.PassCheck=1
        
    def login(self):
        self.baglanti_olustur()
        kullanici_adi = self.ui.line_username.text()
        parola = self.ui.line_pass.text()
        self.cursor.execute("Select *From üyeler where kullanici_adi = ? and parola = ?", (kullanici_adi, parola))
        data = self.cursor.fetchall()
        if len(data) != 0:
            self.process.show()
        else:
            text="Your username or password is wrong"
            self.error(text)
        self.baglanti.close()
    def baglanti_olustur(self):
        self.baglanti = sqlite3.connect("users.db")
        self.cursor = self.baglanti.cursor()
        self.cursor.execute("Create Table If not exists üyeler (email TEXT,kullanici_adi TEXT,parola TEXT)")
        self.baglanti.commit()   
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()