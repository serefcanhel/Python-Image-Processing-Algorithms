# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:28:00 2021

@author: SEREF
"""
from register_python import Ui_MainWindow
from PyQt5.QtWidgets import*
import sqlite3
import time

class register(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("REGISTER")
        
        self.ui.pBut_register.clicked.connect(self.register)
        
    def register(self):
        if(int(self.ui.line_mail.text()=="") | int(self.ui.line_username.text()=="") | int(self.ui.line_pass.text()=="")):
            text='PLEASE FILL ALL SECTIONS !!'
            self.error(text)
        else:
            baglanti = sqlite3.connect("users.db")
            self.cursor = baglanti.cursor()
            self.cursor.execute("Create Table If not exists üyeler (email TEXT,kullanici_adi TEXT,parola TEXT)")
            email = self.ui.line_mail.text()
            kullanici_adi = self.ui.line_username.text()
            parola = self.ui.line_pass.text()
            self.cursor.execute("Select *From üyeler where email= ? and kullanici_adi = ? and parola = ?",(email, kullanici_adi, parola))
            data = self.cursor.fetchall()
            if len(email) == 0 or len(kullanici_adi) == 0 or len(parola) == 0:
                time.sleep(1)
                self.label_message.setStyleSheet("background-color: rgb(216, 216, 216);")
                self.label_message.setText("Incorrect entry")
            else:
                if len(data) == 0:
                    if(parola == self.ui.line_pass_2.text):                
                        self.cursor.execute("INSERT into üyeler(email,kullanici_adi,parola) VALUES(?,?,?) ",(email, kullanici_adi, parola))
                        baglanti.commit()
                        time.sleep(1)
                        text='Your account has created.'
                        self.error(text)
                    else:
                        text='PASSWORDS ARE NOT MATCHED'
                        self.error(text)
                else:
                    time.sleep(1)
                    text='You already have an account'
                    self.error(text)
            baglanti.close()
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()