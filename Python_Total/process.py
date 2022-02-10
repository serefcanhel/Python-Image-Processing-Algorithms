# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:55:03 2021

@author: Zeynep
"""

from process_python import Ui_MainWindow
from PyQt5.QtWidgets import*
from PyQt5.QtGui import QIcon
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
from PyQt5 import QtGui
import math
import matplotlib.pyplot as plt
import cv2
import random
from on_mouse import *
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


class Process(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("PROCESS")
        
        self.ui.cb_Menu.currentTextChanged.connect(self.MenuChanged)
        self.ui.pb_Choose_2.clicked.connect(self.ChooseBlurImage)
        self.ui.pb_Choose_3.clicked.connect(self.ChooseTransformationImage)
        self.ui.pb_Choose_4.clicked.connect(self.ChooseConversionImage)
        self.ui.pb_Choose_5.clicked.connect(self.ChooseHistogramImage)
        self.ui.pb_Choose_6.clicked.connect(self.ChooseFilterImage)
        self.ui.pb_Choose_7.clicked.connect(self.ChooseSkeletonImage)
        self.ui.pb_Choose_8.clicked.connect(self.ChooseHighPassImage)
        self.ui.pb_Choose_9.clicked.connect(self.ChooseMoireFilterImage)
        self.ui.pb_Choose_10.clicked.connect(self.ChooseNoiseImage)
        self.ui.pb_Choose_12.clicked.connect(self.RemoveNoise)
        self.ui.pb_Choose_13.clicked.connect(self.ChooseIm)
        self.ui.pb_Choose_15.clicked.connect(self.Choose)
        self.ui.pb_Blur.clicked.connect(self.BlurImage)
        self.ui.pb_ApplyTransform.clicked.connect(self.Transformations)
        self.ui.pb_Convert.clicked.connect(self.ConvertChanged)
        self.ui.pb_Applyhistog.clicked.connect(self.Histogram)
        self.ui.pb_ApplyFilter.clicked.connect(self.Filter)
        self.ui.pb_HighPassFilter.clicked.connect(self.ApplyHighPassFilter)
        self.ui.pb_Threshholding.clicked.connect(self.ApplyThresholding)
        self.ui.pb_Choose_11.clicked.connect(self.AddNoise)
        self.ui.pb_HighPassFilter_2.clicked.connect(self.ApplyMoirePattern)
        self.ui.cb_Transformation.currentTextChanged.connect(self.TransformationChanged)
        self.ui.cb_Filter.currentTextChanged.connect(self.FilterChanged)
        self.ui.cb_HMW3.currentTextChanged.connect(self.HMW3Changed)
        self.ui.pb_ApplyFilter_2.clicked.connect(self.SkeletonChanged)
        self.ui.highValue_2.valueChanged.connect(self.lowLevelChanged)
        self.ui.highValue.valueChanged.connect(self.highLevelChanged)
        self.ui.sp_Filter.valueChanged.connect(self.filterValueChanged)
        self.ui.cb_Menu_2.currentTextChanged.connect(self.NoiseChanged)
        self.ui.cb_Menu_3.currentTextChanged.connect(self.FilterChanged)
        self.ui.cb_Menu_4.currentTextChanged.connect(self.Changed)
        self.ui.pb_Choose_14.clicked.connect(self.Apply)
        self.ui.pb_Choose_16.clicked.connect(self.ApplyOtsu)
        self.ui.pb_Choose_17.clicked.connect(self.ApplyRegionGrowing)
        
    def MenuChanged(self):
        if(self.ui.cb_Menu.currentText()=="BLUR IMAGE"):
            self.ui.stackedWidget.setCurrentIndex(1)
        elif(self.ui.cb_Menu.currentText()=="TRANSFORMATIONS"):
            self.ui.stackedWidget.setCurrentIndex(3)
        elif(self.ui.cb_Menu.currentText()=="CONVERSIONS"):
            self.ui.stackedWidget.setCurrentIndex(4)
        elif(self.ui.cb_Menu.currentText()=="HISTOGRAM"):
            self.ui.stackedWidget.setCurrentIndex(5)
        elif(self.ui.cb_Menu.currentText()=="FILTERING"):
            self.ui.stackedWidget.setCurrentIndex(7)
        elif(self.ui.cb_Menu.currentText()=="HISTOGRAM GRAPHS"):
            self.ui.stackedWidget.setCurrentIndex(6)
        elif(self.ui.cb_Menu.currentText()=="SKELETON"):
            self.ui.stackedWidget.setCurrentIndex(2)
        elif(self.ui.cb_Menu.currentText()=="High Pass Filtering"):
            self.ui.stackedWidget.setCurrentIndex(8)
        elif(self.ui.cb_Menu.currentText()=="Moire Pattern"):
            self.ui.stackedWidget.setCurrentIndex(9)
        elif(self.ui.cb_Menu.currentText()=="Noise&Filter"):
            self.ui.stackedWidget.setCurrentIndex(10) 
        elif(self.ui.cb_Menu.currentText()=="Noise Reduction"):
            self.ui.stackedWidget.setCurrentIndex(11)
        elif(self.ui.cb_Menu.currentText()=="OTSU&Region"):
            self.ui.stackedWidget.setCurrentIndex(12)
    
    def TransformationChanged(self):
        if(self.ui.cb_Transformation.currentText()=="Resize Image"):
            self.ui.le_Trans1.setText("Row Size")
            self.ui.le_Trans2.setText("Column Size")
        elif(self.ui.cb_Transformation.currentText()=="Crop Image"):
            self.ui.le_Trans1.setText("Left Point")
            self.ui.le_Trans2.setText("Right Point")
            self.ui.le_Trans3.setText("Top Point")
            self.ui.le_Trans4.setText("Bottom Point")
    
    def HMW3Changed(self):
        if(self.ui.cb_HMW3.currentText()=="Transfer Function"):
            self.ui.le_Hist1.setText("High Level")
            self.ui.le_Hist2.setText("Low Level")
        elif(self.ui.cb_HMW3.currentText()=="Add Intensity"):
            self.ui.le_Hist1.setText("Intensity Value")
        elif(self.ui.cb_HMW3.currentText()=="Subtract Intensity"):
            self.ui.le_Hist1.setText("Intensity Value")
            
    def ConvertChanged(self):
        if(self.ui.cb_Convert.currentText()=="RGB ---> GRAY"):
           self.RGBtoGRAY()
        elif(self.ui.cb_Convert.currentText()=="RGB ---> YIQ"):
            self.RGBtoYIQ()
        elif(self.ui.cb_Convert.currentText()=="YIQ ---> RGB"):
            self.YIQtoRGB()
        elif(self.ui.cb_Convert.currentText()=="RGB ---> HSV"):
            self.RGBtoHSV()
        elif(self.ui.cb_Convert.currentText()=="RGB ---> HSI"):
            self.RGBtoHSI()
        elif(self.ui.cb_Convert.currentText()=="RGB ---> LAB"):
            self.RGBtoLAB()
    
    def FilterChanged(self):
        if(self.ui.cb_Filter.currentText()=="Median Filter"):
            self.ui.le_Filter.setText("Kernel Size")
        elif(self.ui.cb_Filter.currentText()=="Minimum Filter"):
            self.ui.le_Filter.setText("Kernel Size")
        elif(self.ui.cb_Filter.currentText()=="Maximum Filter"):
            self.ui.le_Filter.setText("Kernel Size")
        elif(self.ui.cb_Filter.currentText()=="Average Filter"):
            self.ui.le_Filter.setText("Kernel Size")
    
    def NoiseChanged(self):
        if(self.ui.cb_Menu_2.currentText()=="Salt Noise"):
            self.ui.le_Filter_2.setText("Salt Density")
            self.ui.le_Filter_3.setText("Pepper Density")
        elif(self.ui.cb_Menu_2.currentText()=="Pepper Noise"):
            self.ui.le_Filter_2.setText("Salt Density")
            self.ui.le_Filter_3.setText("Pepper Density")
        elif(self.ui.cb_Menu_2.currentText()=="Salt&Pepper Noise"):
            self.ui.le_Filter_2.setText("Salt Density")
            self.ui.le_Filter_3.setText("Pepper Density")
        elif(self.ui.cb_Menu_2.currentText()=="Gaussian Noise"):
            self.ui.le_Filter_2.setText("Mean")
            self.ui.le_Filter_3.setText("Variance")
            
    def FilterChanged(self):
        if(self.ui.cb_Menu_3.currentText()=="Median Filter"):
            self.ui.le_Filter_2.setText("Row Value")
            self.ui.le_Filter_3.setText("Column Value")
        elif(self.ui.cb_Menu_3.currentText()=="Max Filter"):
            self.ui.le_Filter_2.setText("Row Value")
            self.ui.le_Filter_3.setText("Column Value")
        elif(self.ui.cb_Menu_3.currentText()=="Min Filter"):
            self.ui.le_Filter_2.setText("Row Value")
            self.ui.le_Filter_3.setText("Column Value")
        elif(self.ui.cb_Menu_3.currentText()=="Average Filter"):
            self.ui.le_Filter_2.setText("Row Value")
            self.ui.le_Filter_3.setText("Column Value")
        elif(self.ui.cb_Menu_3.currentText()=="Speckle Filter"):
            self.ui.le_Filter_2.setText("Salt Density")
            self.ui.le_Filter_3.setText("Pepper Density")
    
    def Changed(self):
        if(self.ui.cb_Menu_4.currentText()=="Open Image"):
            self.ui.le_Filter_6.setText("Radius:")
        elif(self.ui.cb_Menu_4.currentText()=="Close Image"):
            self.ui.le_Filter_6.setText("Radius:")
        elif(self.ui.cb_Menu_4.currentText()=="Extract Gradient"):
            self.ui.le_Filter_6.setText("Radius:")
        elif(self.ui.cb_Menu_4.currentText()=="Top Hat Transformation"):
            self.ui.le_Filter_6.setText("Radius:")
        elif(self.ui.cb_Menu_4.currentIndex()=="Textual Transformation"):
            self.ui.le_Filter_6.setText("Open Radius:")
            self.ui.le_Filter_8.setText("Close Radius:")
    
    def Transformations(self):
        image_r,image_g, image_b=self.get_rgb_array(self.image)
        if(self.ui.cb_Transformation.currentText()=="Reflection on the X-Axis"):
            image_r=self.ReflectX(image_r)
            image_g=self.ReflectX(image_g)
            image_b=self.ReflectX(image_b)
            self.FinalImage=self.reunite_rgb_image(image_r, image_g, image_b)
            self.ShowBluredImage()
        elif(self.ui.cb_Transformation.currentText()=="Reflection on the Y-Axis"):
            image_r=self.ReflectY(image_r)
            image_g=self.ReflectY(image_g)
            image_b=self.ReflectY(image_b)
            self.FinalImage=self.reunite_rgb_image(image_r, image_g, image_b)
            self.ShowBluredImage()
        elif(self.ui.cb_Transformation.currentText()=="Reflection on the Both Axis"):
            self.ReflectBoth()
        elif(self.ui.cb_Transformation.currentText()=="Resize Image"):
            self.Resize()
        elif(self.ui.cb_Transformation.currentText()=="Crop Image"):
            self.Crop()
    
    def Histogram(self):
        if(self.ui.cb_HMW3.currentText()=="Transfer Function"):
            self.TransferFunction()
        elif(self.ui.cb_HMW3.currentText()=="Add Intensity"):
            self.AddIntensity()
        elif(self.ui.cb_HMW3.currentText()=="Subtract Intensity"):
            self.SubtractIntensity()
        elif(self.ui.cb_HMW3.currentText()=="Histogram Equalization"):
            self.CallHistogramEqualized()
        elif(self.ui.cb_HMW3.currentText()=="Histogram Stretching"):
            return
   
    def Filter(self):
        if(self.ui.cb_Filter.currentText()=="Median Filter"):
            self.MedianFilter()
        elif(self.ui.cb_Filter.currentText()=="Minimum Filter"):
            self.MinimumFilter()
        elif(self.ui.cb_Filter.currentText()=="Maximum Filter"):
            self.MaximumFilter()
        elif(self.ui.cb_Filter.currentText()=="Average Filter"):
            self.AverageFilter()
    
    def SkeletonChanged(self):
        if(self.ui.cb_Filter_2.currentText()=="Laplacian"):
            self.LaplacianFilter()
        elif(self.ui.cb_Filter_2.currentText()=="Sobel"):
            self.applySobelFilter()  
        elif(self.ui.cb_Filter_2.currentText()=="Sharpened Image"):
            self.applySharpened()
        elif(self.ui.cb_Filter_2.currentText()=="Sobel Smooth"):
            self.applySobelSmooth()
        elif(self.ui.cb_Filter_2.currentText()=="Mask Image"):
            self.applyMaskImage()
        elif(self.ui.cb_Filter_2.currentText()=="Image+Mask"):
            self.lastConversion()
        elif(self.ui.cb_Filter_2.currentText()=="Power Low"):
            self.applyPowerLaw()
                
    def lowLevelChanged(self):
        self.ui.le_Lowvalue.setText(str(self.ui.highValue_2.value()))
    def highLevelChanged(self):
        self.ui.le_Highvalue.setText(str(self.ui.highValue.value()))    
    def filterValueChanged(self):
        self.ui.le_Filtersb.setText(str(self.ui.sp_Filter.value()))
    
    def ChooseImage(self):
        self.path = QFileDialog.getOpenFileName(self,"Select Image file to import",""," (*.jpg *.jpeg *.png *.jfif *.bmp *.tif)")[0] 
        self.image = np.array(Image.open(self.path))         
        im = Image.fromarray(self.image)                          
        im.save("resize.jpeg")                                    
        self.resizedImage = Image.open("resize.jpeg")                  
        self.resizedImage = self.resizedImage.resize((512, 512))            
        self.resizedImage.save("resize.jpeg")                          
        self.resizedImage = QtGui.QPixmap("resize.jpeg") 
    def SkeletonImage(self):
        self.path = QFileDialog.getOpenFileName(self,"Select Image file to import",""," (*.jpg *.jpeg *.png *.jfif *.bmp *.tif)")[0] 
        self.skelatonImage = np.array(cv2.imread(self.path, 0))
        im = Image.fromarray(self.skelatonImage)                          
        im.save("resize.jpeg")                                    
        self.resizedImage = Image.open("resize.jpeg")                  
        self.resizedImage = self.resizedImage.resize((512, 512))            
        self.resizedImage.save("resize.jpeg")                          
        self.resizedImage = QtGui.QPixmap("resize.jpeg")  
        
    def ChooseHighImage(self):
        filename = QFileDialog.getOpenFileName(self,'Open File',"C:/Users/Serefcan/Desktop/Seref/ders notları/Image Processing/MATLAB")
        self.imagePath = filename[0]
        pixmap = QtGui.QPixmap(self.imagePath)
        self.ui.l_image_17.setPixmap(QtGui.QPixmap(pixmap)) # load image to label
        self.original_img = self.imagePath
        
    def ChooseIm(self):
        filename = QFileDialog.getOpenFileName(self,'Open File','C:\\Users')
        self.imagePath = filename[0]
        self.image = cv2.imread(self.imagePath)
        pixmap = QtGui.QPixmap(self.imagePath)
        self.ui.l_image_23.setPixmap(QtGui.QPixmap(pixmap))
        self.original_img = self.imagePath
    
    def Choose(self):
        # browse image
        filename = QFileDialog.getOpenFileName(self,'Open File','C:\\Users')
        self.imagePath = filename[0]
        pixmap = QtGui.QPixmap(self.imagePath)
        self.ui.l_image_25.setPixmap(QtGui.QPixmap(pixmap))
        self.original_img = self.imagePath
  
         
    def ChooseBlurImage(self):
        self.ChooseImage()
        self.ui.l_image_4.setPixmap(self.resizedImage)
    def ChooseTransformationImage(self):
        self.ChooseImage()
        self.ui.l_image_5.setPixmap(self.resizedImage)
    def ChooseConversionImage(self):
        self.ChooseImage()
        self.ui.l_image_7.setPixmap(self.resizedImage)             
    def ChooseHistogramImage(self):
        self.ChooseImage()
        self.ui.l_image_9.setPixmap(self.resizedImage) 
    def ChooseFilterImage(self):
        self.ChooseImage()
        self.ui.l_image_11.setPixmap(self.resizedImage)
    def ChooseSkeletonImage(self):
        self.SkeletonImage()
        self.ui.l_image_15.setPixmap(self.resizedImage)
    def ChooseHighPassImage(self):
        self.ChooseImage()
        self.ui.l_image_17.setPixmap(self.resizedImage) 
    def ChooseMoireFilterImage(self):
        self.ChooseImage()
        self.ui.l_image_19.setPixmap(self.resizedImage)
    def ChooseNoiseImage(self):
        self.ChooseImage()
        self.ui.l_image_21.setPixmap(self.resizedImage)

    
    def ShowBluredImage(self):
        im = Image.fromarray(self.FinalImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_6.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_6.setPixmap(foto) 
    
    
    def get_rgb_array(self,image):
        return image[:,:,0], image[:,:,1], image[:,:,2]
    
    def reunite_rgb_image(self,img_r, img_g, img_b):
        shape = (img_r.shape[0], img_r.shape[1], 1)
        reunitedImage = np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape), np.reshape(img_b, shape)), axis=2)
        reunitedImage = np.require(reunitedImage, np.uint8, 'C') 
        return reunitedImage
    
    def BlurImage(self):
        RowSize=self.ui.sb_BlurRow.value()
        ColumnSize=self.ui.sb_BlurColumn.value()
        
        try:
            if self.image.shape[2]==3:
                image_r,image_g, image_b=self.get_rgb_array(self.image)
                image_r = self.reduce(image_r,RowSize,ColumnSize)   
                image_g = self.reduce(image_g,RowSize,ColumnSize)   
                image_b = self.reduce(image_b,RowSize,ColumnSize)   
                
            
                self.bluredImage = self.reunite_rgb_image(image_r, image_g, image_b)
                im = Image.fromarray(self.bluredImage)                         #array to image
                im.save("bluredImage.jpeg")                                          #save image
                foto =QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
                self.ui.l_image_3.setPixmap(foto)                                       
                foto = Image.open("bluredImage.jpeg")                                #get image
                foto = foto.resize((512, 512))                                       #resize image
                foto.save("bluredImage.jpg")                                         #save image
                foto =QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
                self.ui.l_image_3.setPixmap(foto)  
        except:
            self.bluredImage = self.reduce(self.image,RowSize,ColumnSize) 
            self.bluredImage = np.require(self.bluredImage, np.uint8, 'C')          
            im = Image.fromarray(self.final_image_array)                                                          
            im.save("bluredImage.jpeg")                                                         
            foto = QtGui.QPixmap("bluredImage.jpeg")                                                  
            self.ui.l_image_3.setPixmap(foto)
            foto = Image.open("bluredImage.jpeg")                                               
            foto = foto.resize((512, 512))                                                                     
            foto.save("bluredImage.jpg")                                                        
            foto = QtGui.QPixmap("bluredImage.jpg")                                                   
            self.ui.l_image_3.setPixmap(foto)   
    
    def reduce(self,image, rowSize, columnSize):
        a=0
        b=0
        if image.shape[0] % rowSize != 0:
            a = 1
        if image.shape[1] % columnSize != 0:
            b = 1
        blurimage = np.zeros((image.shape[0] ,image.shape[1]))                     
        for i in range(blurimage.shape[0]):
            for j in range(blurimage.shape[1]):
                blurimage[i*rowSize:(i+1)*rowSize,j*columnSize:(j+1)*columnSize] = np.mean(image[i*rowSize:(i+1)*rowSize,j*columnSize:(j+1)*columnSize])    
        
        if b==1:
            for i in range(blurimage.shape[0]):
                blurimage[i*rowSize:(i+1)*rowSize,blurimage.shape[1]-(blurimage.shape[1]%columnSize):blurimage.shape[1]] = np.mean(image[i*rowSize:(i+1)*rowSize,blurimage.shape[1]-(blurimage.shape[1]%columnSize):blurimage.shape[1]]) 
        if a==1:
            for i in range(blurimage.shape[1]):
                blurimage[blurimage.shape[0]-(blurimage.shape[0]%rowSize):blurimage.shape[0],j*columnSize:(j+1)*columnSize] = np.mean(image[blurimage.shape[0]-(blurimage.shape[0]%rowSize):blurimage.shape[0],j*columnSize:(j+1)*columnSize]) 
        
        if a==1 and b==1:
            blurimage[-1,-1] = np.mean(image[blurimage.shape[0]-(blurimage.shape[0]%rowSize):blurimage.shape[0],blurimage.shape[1]-(blurimage.shape[1]%columnSize):blurimage.shape[1]])
        
        return blurimage
                 
    def ReflectX(self,image):
        XImage = np.zeros((self.image.shape[0] , self.image.shape[1]))
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                XImage[image.shape[1]-j-1][i] = image[j][i]
        return XImage
    
    def ReflectY(self,image):
        YImage = np.zeros((self.image.shape[0] , self.image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                YImage[i][image.shape[1]-j-1] = image[i][j]
        return YImage
    
    def ReflectBoth(self):
        image = np.array(Image.open(self.path))/256.0 #CREATE NUMPY ARRAY OF IMAGE
        XImage=image[:, ::-1, :]
        XYImage=XImage[::-1, :,:]
        Im=Image.fromarray((XYImage * 255).astype(np.uint8))
        Im.save("bluredImage.jpg")
        FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_6.setPixmap(QtGui.QPixmap(FinalImage))
    def Resize(self):
        image = np.array(Image.open(self.path))/256.0 #CREATE NUMPY ARRAY OF IMAGE
        RowSize, ColumnSize = image.shape[:2];  
        
        NewSizeRow = self.ui.sb_Trans1.value();
        NewSizeColumn = self.ui.sb_Trans2.value();

        xScale = NewSizeRow/(RowSize-1);
        yScale = NewSizeColumn/(ColumnSize-1);

        ResizedImage = np.zeros([NewSizeRow, NewSizeColumn, 3]);
          
        for i in range(NewSizeRow-1):
           for j in range(NewSizeColumn-1):
               ResizedImage[i + 1, j + 1]= image[1 + int(i / xScale),
                                         1 + int(j / yScale)]
        Im=Image.fromarray((ResizedImage * 255).astype(np.uint8))
        Im.save("bluredImage.jpg")
        FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_6.setPixmap(QtGui.QPixmap(FinalImage))      
    def Crop(self):
        image = np.array(Image.open(self.path))/256.0 #CREATE NUMPY ARRAY OF IMAGE
        image_r,image_g, image_b=self.get_rgb_array(image)
        crop_r=image_r[self.ui.sb_Trans1.value():self.ui.sb_Trans2.value(),self.ui.sb_Trans3.value():self.ui.sb_Trans4.value()] 
        crop_g=image_g[self.ui.sb_Trans1.value():self.ui.sb_Trans2.value(),self.ui.sb_Trans3.value():self.ui.sb_Trans4.value()] 
        crop_b=image_b[self.ui.sb_Trans1.value():self.ui.sb_Trans2.value(),self.ui.sb_Trans3.value():self.ui.sb_Trans4.value()] 
        shape = (crop_r.shape[0], crop_r.shape[1], 1)
        CroppedImage=np.concatenate((np.reshape(crop_r, shape), np.reshape(crop_g, shape), np.reshape(crop_b, shape)), axis=2)
        Im=Image.fromarray((CroppedImage * 255).astype(np.uint8))
        Im.save("bluredImage.jpg")
        FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_6.setPixmap(QtGui.QPixmap(FinalImage))
        
    def RGBtoGRAY(self):
        image_r,image_g, image_b=self.get_rgb_array(self.image)
        GrayImage = 0.2989 * image_r + 0.5870 * image_g + 0.1140 * image_b
        GrayImage = np.require(GrayImage, np.uint8, 'C') 
        im = Image.fromarray(GrayImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)
    
    def RGBtoYIQ(self):
        image_r,image_g, image_b=self.get_rgb_array(self.image)
        YImage=(0.299*image_r + 0.587*image_g + 0.114*image_b)
        IImage = (0.596*image_r - 0.275*image_g - 0.321*image_b)
        QImage = (0.212*image_r - 0.523*image_g + 0.311*image_b)
        shape = (image_r.shape[0], image_r.shape[1], 1)
        YIQImage = np.concatenate((np.reshape(YImage, shape), np.reshape(IImage, shape), np.reshape(QImage, shape)), axis=2)
        YIQImage= np.require(YIQImage, np.uint8, 'C') 
        im = Image.fromarray(YIQImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)  
    
    def YIQtoRGB(self):
        image_Y, image_I, image_Q = self.get_rgb_array(self.image)
        
        RImage = (image_Y + 0.9563*image_I + 0.6210*image_Q)
        GImage = (image_Y - 0.2721*image_I - 0.6474*image_Q)
        BImage = (image_Y - 01.1070*image_I + 1.7046*image_Q)
        shape = (image_Y.shape[0], image_Y.shape[1], 1)
        RGBImage=np.concatenate((np.reshape(RImage, shape), np.reshape(GImage, shape), np.reshape(BImage, shape)), axis=2)
        RGBImage = np.require(RGBImage, np.uint8, 'C') 
        im = Image.fromarray(RGBImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_8.setPixmap(foto)   
        
    def RGBtoHSV(self):
        self.image = self.image.astype('float')
        maxv = np.amax(self.image, axis=2)
        maxc = np.argmax(self.image, axis=2)
        minv = np.amin(self.image, axis=2)
        minc = np.argmin(self.image, axis=2)
    
        hsv = np.zeros(self.image.shape, dtype='float')
        hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
        hsv[maxc == 0, 0] = (((self.image[..., 1] - self.image[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
        hsv[maxc == 1, 0] = (((self.image[..., 2] - self.image[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
        hsv[maxc == 2, 0] = (((self.image[..., 0] - self.image[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
        hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
        hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
        hsv[..., 2] = maxv
        Im=Image.fromarray((hsv * 255).astype(np.uint8))
        Im.save("bluredImage.jpg")
        FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_8.setPixmap(QtGui.QPixmap(FinalImage))
        
    def RGBtoHSI(self):
        image_r,image_g, image_b=self.get_rgb_array(self.image)
        Intensity=np.divide(image_r+image_g+image_b,3)
        minValue = np.minimum(np.minimum(image_r, image_g), image_b)
        Saturation = 1 - (3 / (image_r + image_g + image_b + 0.001) * minValue)
        Hue = np.copy(image_r)
        for i in range(0, image_b.shape[0]):
                for j in range(0, image_b.shape[1]):
                    Hue[i][j] = 0.5 * ((image_r[i][j] - image_g[i][j]) + (image_r[i][j] - image_b[i][j])) / \
                                math.sqrt((image_r[i][j] - image_g[i][j])**2 +
                                        ((image_r[i][j] - image_b[i][j]) * (image_g[i][j] - image_b[i][j])))
                    Hue[i][j] = math.acos(Hue[i][j])

                    if image_b[i][j] <= image_g[i][j]:
                        Hue[i][j] = Hue[i][j]
                    else:
                        Hue[i][j] = ((360 * math.pi) / 180.0) - Hue[i][j]
        HSIImage = cv2.merge((Hue, Saturation, Intensity))
        Im=Image.fromarray((HSIImage * 255).astype(np.uint8))
        Im.save("bluredImage.jpg")
        FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_8.setPixmap(QtGui.QPixmap(FinalImage))   
        
    def TransferFunction(self):
        img_r,img_g,img_b=self.get_rgb_array(self.image)
        
        lowLevel=self.ui.highValue_2.value()
        highLevel=self.ui.highValue.value()
        
        new_img_r = np.array([np.where(i>highLevel, 255, i) for i in img_r])
        new_img_g = np.array([np.where(i>highLevel, 255, i) for i in img_g])
        new_img_b = np.array([np.where(i>highLevel, 255, i) for i in img_b])
        
        new__r = np.array([np.where(i<lowLevel, 0, i) for i in new_img_r])
        new__g = np.array([np.where(i<lowLevel, 0, i) for i in new_img_g])
        new__b = np.array([np.where(i<lowLevel, 0, i) for i in new_img_b])
        
        shape = (new__r.shape[0], new__r.shape[1], 1)
        TransferedImage = np.concatenate((np.reshape(new__r, shape), np.reshape(new__g, shape), np.reshape(new__b, shape)), axis=2)
        TransferedImage = np.require(TransferedImage, np.uint8, 'C') 
        im = Image.fromarray(TransferedImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)                                     #show Blured Image on the label
    
    def AddIntensity(self):
        value = self.ui.highValue.value()
        self.image= np.require(self.image, np.int16, 'C') 
        self.image += value
        img_r, img_g, img_b = self.get_rgb_array(self.image)                 #extract image to colours array
        new_img_r = np.array([np.where(i>255, 255, i) for i in img_r])
        new_img_g = np.array([np.where(i>255, 255, i) for i in img_g])
        new_img_b = np.array([np.where(i>255, 255, i) for i in img_b])
        shape = (new_img_r.shape[0], new_img_r.shape[1], 1)
        ChangedImage = np.concatenate((np.reshape(new_img_r, shape), np.reshape(new_img_g, shape), np.reshape(new_img_b, shape)), axis=2)
        ChangedImage = np.require(ChangedImage, np.uint8, 'C') 
        im = Image.fromarray(ChangedImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto) 
        
    def SubtractIntensity(self):
        value = self.ui.highValue.value()
        self.image= np.require(self.image, np.int16, 'C') 
        self.image += value
        img_r, img_g, img_b = self.get_rgb_array(self.image)                 #extract image to colours array
        new_img_r = np.array([np.where(i<0, 0, i) for i in img_r])
        new_img_g = np.array([np.where(i<0, 0, i) for i in img_g])
        new_img_b = np.array([np.where(i<0, 0, i) for i in img_b])
        shape = (new_img_r.shape[0], new_img_r.shape[1], 1)
        ChangedImage = np.concatenate((np.reshape(new_img_r, shape), np.reshape(new_img_g, shape), np.reshape(new_img_b, shape)), axis=2)
        ChangedImage = np.require(ChangedImage, np.uint8, 'C') 
        im = Image.fromarray(ChangedImage)                         #array to image
        im.save("bluredImage.jpeg")                                          #save image
        foto = QtGui.QPixmap("bluredImage.jpeg")                                   #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)                                       
        foto = Image.open("bluredImage.jpeg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)
        
    def DrawHistogram(self,foto,L):
        histogram,bins=np.histogram(foto,bins=L,range=(0,L))
        return histogram

    def NormaliazedHistogram(self,foto, L):
        histogram = self.DrawHistogram(foto, L)
        plt.plot(histogram)
        plt.savefig("Histogram.png")
        plt.show()
        self.ui.l_image_13.setPixmap(QtGui.QPixmap("Histogram.png"))
        return histogram / foto.size # foto.size = M*N
    
    def CummulativeSum(self,p_r_r):
        p_r_r= iter(p_r_r)
        b = [next(p_r_r)]
        for i in p_r_r:
            b.append(b[-1] + i)
        return np.array(b)
    
    def Histogram_Equalization(self,foto,L):
        p_r_r=self.NormaliazedHistogram(foto,L)
        cummulative=self.CummulativeSum(p_r_r)
        funct=(L-1)*cummulative
        shape=foto.shape
        ravel = foto.ravel() # (800x600) -> 480000
        hist_es_foto = np.zeros_like(ravel)
        for i, pixel in enumerate(ravel):
            hist_es_foto[i] = funct[pixel]
        return hist_es_foto.reshape(shape).astype(np.uint8)
    
    def CallHistogramEqualized(self):
        L=2**8
        EuqalizatedImage=self.Histogram_Equalization(self.image,L)
        Im=Image.fromarray((EuqalizatedImage ))
        Im.save("bluredImage.jpg")
        self.FinalImage=QtGui.QPixmap("bluredImage.jpg")
        self.ui.l_image_10.setPixmap(QtGui.QPixmap("bluredImage.jpg"))
        EuqalizatedImage=self.DrawHistogram(EuqalizatedImage,L)
        plt.plot(EuqalizatedImage)
        plt.savefig("Histogram.png")
        plt.show()
        foto = Image.open("bluredImage.jpg")                                #get image
        foto = foto.resize((512, 512))                                       #resize image
        foto.save("bluredImage.jpg")                                         #save image
        foto = QtGui.QPixmap("bluredImage.jpg")                                    #convert image to PixMap
        self.ui.l_image_10.setPixmap(foto)
        self.ui.l_image_14.setPixmap(QtGui.QPixmap("Histogram.png"))
    
    def MedianFilter(self,row,column):
        MedianFilteredImage = np.zeros((self.NoisedImage.shape[0] ,self.NoisedImage.shape[1])) 
        for i in range(self.NoisedImage.shape[0]-row-1):
            for j in range(self.NoisedImage.shape[1]-column-1):
                MedianFilteredImage[i+1][j+1] = np.median(self.NoisedImage[i:i+row,j:j+column])
        MedianFilteredImage = np.require(MedianFilteredImage, np.uint8, 'C') 
        FilteredImage = np.require(MedianFilteredImage, np.uint8, 'C') 
        im = Image.fromarray(FilteredImage)                                #array to image
        im.save("bluredImage.tif")                                            #save image
        foto = QtGui.QPixmap("bluredImage.tif")                                     #convert image to PixMap    
        self.ui.l_image_22.setPixmap(foto)  
        
    def MinimumFilter(self,row,column):
        MinimumFilteredImage = np.zeros((self.NoisedImage.shape[0] ,self.NoisedImage.shape[1])) 
        
        for i in range(self.NoisedImage.shape[0]-row-1):
            for j in range(self.NoisedImage.shape[1]-column-1):
                MinimumFilteredImage[i+1][j+1] = np.amin(self.NoisedImage[i:i+row,j:j+column])
        MinimumFilteredImage = np.require(MinimumFilteredImage, np.uint8, 'C') 
        FilteredImage = np.require(MinimumFilteredImage, np.uint8, 'C') 
        im = Image.fromarray(FilteredImage)                                #array to image
        im.save("bluredImage.tif")                                            #save image
        foto = QtGui.QPixmap("bluredImage.tif")                                     #convert image to PixMap    
        self.ui.l_image_22.setPixmap(foto)
    
    def MaximumFilter(self,row,column):
        MaximumFilteredImage = np.zeros((self.NoisedImage.shape[0] ,self.NoisedImage.shape[1])) 
        
        for i in range(self.NoisedImage.shape[0]-row-1):
            for j in range(self.NoisedImage.shape[1]-column-1):
                MaximumFilteredImage[i+1][j+1] = np.amax(self.NoisedImage[i:i+row,j:j+column])
        MaximumFilteredImage = np.require(MaximumFilteredImage, np.uint8, 'C') 
        FilteredImage = np.require(MaximumFilteredImage, np.uint8, 'C') 
        im = Image.fromarray(FilteredImage)                                #array to image
        im.save("bluredImage.tif")                                            #save image
        foto = QtGui.QPixmap("bluredImage.tif")                                     #convert image to PixMap    
        self.ui.l_image_22.setPixmap(foto)
        
    def AverageFilter(self,row,column):
        AverageFilteredImage = np.zeros((self.image.shape[0] ,self.image.shape[1])) 
        
        for i in range(self.image.shape[0]-row-1):
            for j in range(self.image.shape[1]-column-1):
                AverageFilteredImage[i+1][j+1] = np.mean(self.image[i:i+row,j:j+column])
        AverageFilteredImage = np.require(AverageFilteredImage, np.uint8, 'C') 
        FilteredImage = np.require(AverageFilteredImage, np.uint8, 'C') 
        im = Image.fromarray(FilteredImage)                                #array to image
        im.save("bluredImage.tif")                                            #save image
        foto = QtGui.QPixmap("bluredImage.tif")                                     #convert image to PixMap    
        self.ui.l_image_22.setPixmap(foto)
        
    def ApplyLaplace(self,image):
        F1 = [[0,1,0],[1,-4, 1],[0,1,0]];

        filteredImage = np.zeros((image.shape[0] ,image.shape[1])) 
        
        for i in range(image.shape[0]-2):
            for j in range(image.shape[1]-2):
                dot_product = np.dot(F1,image[i:i+3,j:j+3])
                summationResult = sum(map(sum, dot_product))
                filteredImage[i+1][j+1] = summationResult

        filteredImage = cv2.Laplacian(image, cv2.CV_64F)
     
        return filteredImage
        
    def LaplacianFilter(self):
        LaplacianFilteredImage = self.ApplyLaplace(self.skelatonImage)
        LaplacianFilteredImage = (((LaplacianFilteredImage - LaplacianFilteredImage.min()) / (LaplacianFilteredImage.max() - LaplacianFilteredImage.min())) * 255.9).astype(np.uint8)
        im = Image.fromarray(LaplacianFilteredImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage)
    
    def sobelfilter(self,image):
        x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        [RowSize, ColumSize] = np.shape(image)  
        SobelFilteredImage = np.zeros(shape=(RowSize, ColumSize))  
        
        
        for i in range(RowSize - 2):
            for j in range(ColumSize - 2):
                Newx = np.sum(np.multiply(x, image[i:i + 3, j:j + 3]))  
                Newy = np.sum(np.multiply(y, image[i:i + 3, j:j + 3]))  
                SobelFilteredImage[i + 1, j + 1] = np.sqrt(Newx ** 2 + Newy ** 2)  
        
        SobelFilteredImage = np.require(SobelFilteredImage, np.uint8, 'C')
        return SobelFilteredImage
    
    def applySobelFilter(self):
        SobelFilteredImage = self.sobelfilter(self.skelatonImage)
        im = Image.fromarray(SobelFilteredImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage)   

    def applySharpened(self):
        LaplacianFilteredImage = self.ApplyLaplace(self.skelatonImage)
        LaplacianFilteredImage  = np.uint8(np.absolute((LaplacianFilteredImage )))
        SharpenedImage = cv2.add(self.skelatonImage, LaplacianFilteredImage )
        im = Image.fromarray(SharpenedImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage)          
    
    def SobelSmoothFilter(self,array):
        SobeLSmooth = cv2.blur(array, (5, 5))
        return SobeLSmooth
    
    def applySobelSmooth(self):
        finalArray = self.sobelfilter(self.skelatonImage)
        finalArray = self.SobelSmoothFilter(finalArray)
        im = Image.fromarray(finalArray) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage)  
        
    def applyMaskImage(self):
        finalArray = self.sobelfilter(self.skelatonImage)
        blurSobel = self.SobelSmoothFilter(finalArray)
        
        LaplacianFiltered = self.ApplyLaplace(self.skelatonImage)
        LaplacianFiltered = np.uint8(np.absolute((LaplacianFiltered)))
        shapImg = cv2.add(self.skelatonImage, LaplacianFiltered)
        
        maskedImage= cv2.bitwise_and(blurSobel, shapImg)
        
        im = Image.fromarray(maskedImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage) 
    
    def lastConversion(self):
        finalArray = self.sobelfilter(self.skelatonImage)
        blurSobel = self.SobelSmoothFilter(finalArray)
        
        LaplacianFiltered = self.ApplyLaplace(self.skelatonImage)
        LaplacianFiltered = np.uint8(np.absolute((LaplacianFiltered)))
        shapImg = cv2.add(self.skelatonImage, LaplacianFiltered)
        
        maskedImage= cv2.bitwise_and(blurSobel, shapImg)
        procesG = cv2.add(self.skelatonImage, maskedImage)
        im = Image.fromarray(procesG) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage) 
    
    def applyPowerLaw(self):
        finalArray = self.sobelfilter(self.skelatonImage)
        blurSobel = self.SobelSmoothFilter(finalArray)
        
        LaplacianFiltered = self.ApplyLaplace(self.skelatonImage)
        LaplacianFiltered = np.uint8(np.absolute((LaplacianFiltered)))
        shapImg = cv2.add(self.skelatonImage, LaplacianFiltered)
        
        maskedImage= cv2.bitwise_and(blurSobel, shapImg)
        procesG = cv2.add(self.skelatonImage, maskedImage)
        powerImg = np.array(255 * (procesG / 255) ** 0.5, dtype='uint8')
        im = Image.fromarray(powerImg) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_16.setPixmap(resizedImage)
    
    def ButterworthHighPassFilter(self,shape):

        CuttofFrequency = 50  
        Order= 4  
        RowValue, ColumnValue= shape
        Mask = np.zeros((RowValue, ColumnValue))
        CenterRow, CenterColumn = int(RowValue / 2), int(ColumnValue / 2)
        for i in range(RowValue):
            for j in range(ColumnValue):
                Distance = math.sqrt((i - CenterRow) ** 2 + (j - CenterColumn) ** 2)
                if Distance == 0:
                    Mask[i, j] = 0
                else:
                    Mask[i, j] = 1 / (1 + (CuttofFrequency / Distance) ** (2 * Order))
        return Mask
        
    def ApplyHighPassFilter(self):
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(image) # 1. take fourier transform of image 
        fftShift = np.fft.fftshift(fft) # 2. shift the fourier tranformed image
        mask = self.get_butterworth_high_pass_filter(np.shape(image)) # 3. call high pass filter
        filtered_image = np.multiply(mask, fftShift) # 4. multiply high pass filter with shifted original image
        shift_ifft = np.fft.ifftshift(filtered_image) # 5. compute the inverse shift
        ifft = np.uint8(np.real(np.fft.ifft2(shift_ifft))) # 6. compute the inverse fourier transform
        mag = np.abs(np.fft.ifft2(shift_ifft)) # 7. compute the magnitude
        filtered_image = np.uint8(mag) # 8. compute the filtered image
        self.thresholding_image = np.uint8(ifft) # 9. compute final thresholding_image

        
        self.saved_filtered_image = 'filtered_image.jpg'
        self.ui.l_image_18.setPixmap(QtGui.QPixmap(self.saved_filtered_image))

        
    def get_butterworth_high_pass_filter(self,shape):
        d0 = 3  # cut off frequency
        n =4 # order
        rows, columns = shape
        mask = np.zeros((rows, columns))
        mid_R, mid_C = int(rows / 2), int(columns / 2) # center points of rows and columns
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2) # Calculate Euclidean distance
                if d == 0: # to prevent zero division error
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1 / (1 + (d0 / d) ** (2 * n)) # Compute high pass filter by applying formula
        return mask
        
    def ApplyThresholding(self):
        self.saved_thresholding_image = 'thresholding_image.jpg'
        cv2.imwrite(self.saved_thresholding_image, self.thresholding_image)
        
        self.ui.l_image_18.setPixmap(QtGui.QPixmap(self.saved_thresholding_image))


    def NotchFilter(self,ImageShape):
        CuttofFrequency = 3
        n = 4
        W = 100
        RowValue, ColumnValue = ImageShape
        mask = np.zeros((RowValue, ColumnValue))
        # center points of rows and columns
        CenterRow, CenterColumn = int(RowValue / 2), int(ColumnValue / 2)

        for i in range(RowValue):
            for j in range(ColumnValue):
                # Calculate Euclidean distance
                dist = math.sqrt((i - CenterRow) ** 2 + (j - CenterColumn) ** 2)
                # to prevent zero division error
                if dist == 0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 1 - math.exp(-1 * ((dist ** 2 - CuttofFrequency ** 2) / (dist * W)) ** 2)
        return mask
        
    def ApplyMoirePattern(self):
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        FourierTransformedImage = np.fft.fft2(image)
        ShiftedImage = np.fft.fftshift(FourierTransformedImage)
        SpectrumofImage = np.log(np.abs(ShiftedImage))
        dft = np.uint8(SpectrumofImage)
        Mask = self.NotchFilter(np.shape(image))
        NotchFilteredImage = np.multiply(Mask, FourierTransformedImage)
        InverseTransformedImage = np.fft.ifftshift(NotchFilteredImage)
        ifft = np.uint8(np.real(np.fft.ifft2(InverseTransformedImage)))
        InverseImage = np.abs(np.fft.ifft2(InverseTransformedImage))
        NotchFilteredImage = np.uint8(InverseImage)
        
        im = Image.fromarray(NotchFilteredImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_20.setPixmap(resizedImage)
        
    def AddSaltNoise(self,pepper,salt):
        output = np.zeros(self.image.shape,np.uint8)
        pepper=pepper/100
        salt=salt/100
        thres = 1 - (salt+pepper)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < salt:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        return output
    
    def AddPepperNoise(self,pepper,salt):
        output = np.zeros(self.image.shape,np.uint8)
        pepper=pepper/100
        salt=salt/100
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < pepper:
                    output[i][j] = 0
                else:
                    output[i][j] = self.image[i][j]
        return output
    
    def AddSaltPepperNoise(self,pepper,salt):
        output = np.zeros(self.image.shape,np.uint8)
        pepper=pepper/100
        salt=salt/100
        thres = 1 - (salt+pepper)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < salt:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        return output
    
    def AddGaussianNoise(self):
        row,col=self.image.shape
        mean = int(self.ui.le_Filter_4.text())
        var = int(self.ui.le_Filter_5.text())
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (row, col)) # np.zeros((224, 224), np.float32)
        
        noisy_image = np.zeros(self.image.shape, np.float32)
        noisy_image = self.image + gaussian
        return noisy_image
    
    
    def AddPeriodicNoise(self):
        gauss = np.random.normal(0,1,self.image.size)
        gauss = gauss.reshape(self.image.shape[0],self.image.shape[1],self.image.shape[2]).astype('uint8')
        noise = self.image + self.image * gauss
        return noise
   
    def Speckle(self):
        img_mean = uniform_filter(self.NoisedImage, (20, 20))
        img_sqr_mean = uniform_filter(self.NoisedImage**2, (20, 20))
        img_variance = img_sqr_mean - img_mean**2
    
        overall_variance = variance(self.NoisedImage)
    
        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (self.NoisedImage - img_mean)
        return img_output
    
    
    def AddNoise(self):
        if(self.ui.cb_Menu_2.currentText()=="Salt Noise"):
            self.ui.line_factor_22.setText("Salt Noised Image")
            Pepper=int(self.ui.le_Filter_4.text())
            Salt=int(self.ui.le_Filter_5.text())
            self.NoisedImage=self.AddSaltNoise(Pepper, Salt)
            im = Image.fromarray(self.NoisedImage) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            resizedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(resizedImage)
        elif(self.ui.cb_Menu_2.currentText()=="Pepper Noise"):
            self.ui.line_factor_22.setText("Pepper Noised Image")
            Salt=int(self.ui.le_Filter_4.text())
            Pepper=int(self.ui.le_Filter_5.text())
            self.NoisedImage=self.AddPepperNoise(Pepper, Salt)
            im = Image.fromarray(self.NoisedImage) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            resizedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(resizedImage)
        elif(self.ui.cb_Menu_2.currentText()=="Gaussian Noise"):
            self.ui.line_factor_22.setText("Gaussian Noised Image")
            self.NoisedImage=self.AddGaussianNoise()
            final_image_array= np.require(self.NoisedImage, np.uint8, 'C') 
            im = Image.fromarray(final_image_array) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            self.NoisedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(self.NoisedImage)
        elif(self.ui.cb_Menu_2.currentText()=="Speckle Noise"):
            self.ui.line_factor_22.setText("Speckle Noised Image")
            self.NoisedImage=self.AddPeriodicNoise()
            im = Image.fromarray(self.NoisedImage) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            resizedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(resizedImage)
        elif(self.ui.cb_Menu_2.currentText()=="Salt&Pepper Noise"):
            self.ui.line_factor_22.setText("Salt&Pepper Noised Image")
            Salt=int(self.ui.le_Filter_4.text())
            Pepper=int(self.ui.le_Filter_5.text())
            self.NoisedImage=self.AddSaltPepperNoise(Pepper,Salt)
            im = Image.fromarray(self.NoisedImage) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            resizedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(resizedImage)
            
    def RemoveNoise(self):
        if(self.ui.cb_Menu_3.currentText()=="Median Filter"):
            row=int(self.ui.le_Filter_4.text())
            column=int(self.ui.le_Filter_5.text())
            self.MedianFilter(row, column)
            
        elif(self.ui.cb_Menu_3.currentText()=="Max Filter"):
            row=int(self.ui.le_Filter_4.text())
            column=int(self.ui.le_Filter_5.text())
            self.MaximumFilter(row, column)           
        elif(self.ui.cb_Menu_3.currentText()=="Min Filter"):
            row=int(self.ui.le_Filter_4.text())
            column=int(self.ui.le_Filter_5.text())
            self.MinimumFilter(row, column)           
        elif(self.ui.cb_Menu_3.currentText()=="Average Filter"):
            row=int(self.ui.le_Filter_4.text())
            column=int(self.ui.le_Filter_5.text())
            self.AverageFilter(row, column)            
        elif(self.ui.cb_Menu_3.currentText()=="Speckle Filter"):

            a=self.Speckle()
            im = Image.fromarray(a) 
            im.save("blured.jpeg")                                    #save image 
            resizedImage = Image.open("blured.jpeg")                  
            resizedImage = resizedImage.resize((512, 512))            
            resizedImage.save("blured.jpeg")                          
            resizedImage = QtGui.QPixmap("blured.jpeg")                     
            self.ui.l_image_22.setPixmap(resizedImage)
    
    def OpenImage(self):
        radius=int(self.ui.le_Filter_7.text())
        kernel = np.ones((radius,radius),np.uint8)
        OpenedImage = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        im = Image.fromarray(OpenedImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_24.setPixmap(resizedImage)
        self.ui.line_factor_24.setText("Opened Image")
    
    def CloseImage(self):
        radius=int(self.ui.le_Filter_7.text())
        kernel=np.ones((radius,radius),np.uint8)
        ClosedImage=cv2.morphologyEx(self.image,cv2.MORPH_CLOSE,kernel)
        im = Image.fromarray(ClosedImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_24.setPixmap(resizedImage)
        self.ui.line_factor_24.setText("Closed Image")
        
    def ExtractGradient(self):
        radius=int(self.ui.le_Filter_7.text())
        kernel=np.ones((radius,radius),np.uint8)
        ExtractGradientImage=cv2.morphologyEx(self.image,cv2.MORPH_GRADIENT,kernel)
        im = Image.fromarray(ExtractGradientImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_24.setPixmap(resizedImage)
        self.ui.line_factor_24.setText("Extracted Image")
    
    def TopHatTransformation(self):
        radius=int(self.ui.le_Filter_7.text())
        kernel=np.ones((radius,radius),np.uint8)
        TransformatedImage=cv2.morphologyEx(self.image,cv2.MORPH_TOPHAT,kernel)
        im = Image.fromarray(TransformatedImage) 
        im.save("blured.jpeg")                                    #save image 
        resizedImage = Image.open("blured.jpeg")                  
        resizedImage = resizedImage.resize((512, 512))            
        resizedImage.save("blured.jpeg")                          
        resizedImage = QtGui.QPixmap("blured.jpeg")                     
        self.ui.l_image_24.setPixmap(resizedImage)
        self.ui.line_factor_24.setText("Transformed Image")
    
    def TextualTransformation(self):
        OpenRadius=int(self.ui.le_Filter_7.text())
        CloseRadius=int(self.ui.le_Filter_9.text())
        
        
    def Apply(self):
        if(self.ui.cb_Menu_4.currentText()=="Open Image"):
            self.OpenImage()
        elif(self.ui.cb_Menu_4.currentText()=="Close Image"):
            self.CloseImage()
        elif(self.ui.cb_Menu_4.currentText()=="Extract Gradient"):
            self.ExtractGradient()
        elif(self.ui.cb_Menu_4.currentText()=="Top Hat Transformation"):
            self.TopHatTransformation()
        elif(self.ui.cb_Menu_4.currentText()=="Textual Segmentation"):
            self.TextualTransformation()
    
    def OTSUMethod(self,image):
        OTSUFilteredImage = image.copy()

        histogram = np.zeros(256)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                histogram[image[i, j]] += 1

        pixel_number = image.shape[0] * image.shape[1]

        bins = np.array(range(0, 256))
        final_thresh = -1
        final_value = -1
        for t in bins[1:-1]:
            Wb = np.sum(histogram[:t]) / pixel_number
            Wf = np.sum(histogram[t:]) / pixel_number 

            mub = np.mean(histogram[:t])
            muf = np.mean(histogram[t:])
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value

        OTSUFilteredImage[image > final_thresh] = 255
        OTSUFilteredImage[image < final_thresh] = 0
        return OTSUFilteredImage        
    
    def ApplyOtsu(self):
        image = cv2.imread(self.original_img)
        FilteredImage = self.OTSUMethod(image)

        self.saved_img = 'OtsuImage.jpg'
        cv2.imwrite(self.saved_img,FilteredImage)
        self.ui.line_factor_26.setText("OTSU Thresholded Image")
        self.ui.l_image_26.setPixmap(QtGui.QPixmap(self.saved_img))
    
    def mousePressEvent(self, event):
        global Mouse_X
        global Mouse_Y
        "If you press the mouse on interface this function" \
        "will get the x and y coordinates to use in the " \
        "function of region growing as seed."
        try:
            self.Mouse_X = event.x()
            self.Mouse_Y = event.y()
            self.main_window.label_5.setText(" X:{}  Y:{}".format(self.Mouse_X, self.Mouse_Y))
            # self.lastPoint = self.GUI.label_2.mapFromParent(event .pos())
            self.lastPoint = self.label.mapFromParent(event.pos())
        except Exception as msg:
            # logging.error('Error Update_work: ' + str(msg))
            pass
    
    def get_region(self,x, y, shape):
        """get 8 edges to use in region growing function"""
        out = []
        maxx = shape[1] - 1
        maxy = shape[0] - 1

        # top left
        out.append((min(max(x - 1, 0), maxx), min(max(y - 1, 0), maxy)))
        # top center
        out.append((x, min(max(y - 1, 0), maxy)))
        # top right
        out.append((min(max(x + 1, 0), maxx), min(max(y - 1, 0), maxy)))
        # left
        out.append((min(max(x - 1, 0), maxx), y))
        # right
        out.append((min(max(x + 1, 0), maxx), y))
        # bottom left
        out.append((min(max(x - 1, 0), maxx), min(max(y + 1, 0), maxy)))
        # bottom center
        out.append((x, min(max(y + 1, 0), maxy)))
        # bottom right
        out.append((min(max(x + 1, 0), maxx), min(max(y + 1, 0), maxy)))

        return out

    def region_growing_function(self,image,seed):
        list = []
        region_growing_image = np.zeros_like(image)
        list.append((seed[0], seed[1]))
        print(len(list))

        region = []
        while(len(list) > 0):
            pix = list[0]
            region_growing_image[pix[0], pix[1]] = 255
            for i in self.get_region(pix[0], pix[1], image.shape):
                if image[i[0], i[1]] != 0:
                    region_growing_image[i[0], i[1]] = 255
                    if not i in region:
                        list.append(i)
                    region.append(i)
            list.pop(0)

        return region_growing_image

    def ApplyRegionGrowing(self):
        self.ui.l_image_26.setPixmap(QtGui.QPixmap('RegionGrowingImage.jpg'))
        image = cv2.imread(self.original_img, 0)
        img = self.OTSUMethod(image)
        seed = (self.Mouse_X, self.Mouse_Y)
        out = self.region_growing_function(img, seed)
        self.saved_img = 'RegionGrowingImage.jpg'
        cv2.imwrite(self.saved_img, out)
        self.ui.line_factor_26.setText("Region Growing Operation")
        self.ui.l_image_26.setPixmap(QtGui.QPixmap(self.saved_img))


            
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()