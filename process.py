#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:57:29 2018

@author: david
"""

import os
import cv2 as cv 

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

imageFile = "./data"
transferFile = "./transfer"
if not os.path.exists(transferFile):
    os.mkdir(transferFile)
for folders in os.listdir(imageFile):
    path = imageFile+'/'+folders
    if not os.path.exists(transferFile+'/'+folders):
        os.mkdir(transferFile+'/'+folders)
    for emotions in os.listdir(path):
        path_e = path+'/'+str(emotions)
        if not os.path.exists(transferFile+'/'+folders+'/'+emotions):
            os.mkdir(transferFile+'/'+folders+'/'+str(emotions))
        for pictures in os.listdir(path_e):
            picPath = path_e+'/'+str(pictures)
            image = cv.imread(picPath,1)
            try: 
                rows,cols,dim = image.shape
            except AttributeError:
                print("something went wrong: "+picPath)
                continue
            if dim ==1:
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                wp = int(w*1.2)
                hp = int(h*1.2)
                x = int(x-(wp-w)/2.)
                y = int(y-(hp-h)/2.)
                if x<0:
                    x = 0 
                if y <0:
                    y = 0
                crop_img = image[y:y+hp, x:x+wp]

                #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                dest = (str(transferFile)+'/'+str(folders)+'/'+str(emotions)+'/'+str(pictures))
                cv.imwrite(dest,crop_img)