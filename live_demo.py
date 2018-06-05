#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:08:58 2018

@author: david
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models.resnet import model_urls as model_url_resnet
from torchvision.models.alexnet import model_urls as model_url_alexnet
from torchvision.models.vgg import model_urls as model_url_vgg
import argparse
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
parser = argparse.ArgumentParser(description='Emotion Detection Demo')
parser.add_argument('--useGPU_f', action='store_true', default=False, help='Flag to use GPU (STORE_FALSE)(default: False)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['AlexNet', 'ResNet', 'VGG'], help="net model(default:AlexNet)")
parser.add_argument("--dataset", default='Emotions', const='Emotions',nargs='?', choices=['Emotions', 'ImageNet'], help="Dataset (default:Emotions)")
parser.add_argument('--c', '--NumClasses', action='store', default=8, type=float, help='number of classes (default: 8)')
arg = parser.parse_args()
MODEL_PATH = './data/old_model_'+arg.net+'_'+str(arg.dataset)+'.pt'
CLASS_NUM = arg.numClasses
FONT = cv2.FONT_HERSHEY_SIMPLEX
classNames = ['anger','contempt','disgust','fear','happy','neutral','sadness','surprise']


#import the model here
if arg.net == 'AlexNet':
	model_url_alexnet['alexnet'] = model_url_alexnet['alexnet'].replace('https://', 'http://')
	model = models.alexnet(pretrained=arg.preTrained_f)
elif arg.net == 'ResNet':
	model_url_resnet['resnet18'] = model_url_resnet['resnet18'].replace('https://', 'http://')
	model = models.resnet18(pretrained=arg.preTrained_f)
elif arg.net == 'VGG':
	model_url_vgg['vgg16'] = model_url_vgg['vgg16'].replace('https://', 'http://')
	model = models.vgg16(pretrained=arg.preTrained_f)
preprocess = transforms.Compose([
			transforms.Resize(240),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])            
if arg.dataset == 'Emotions':
	if arg.net == 'ResNet':
		model.fc= nn.Linear(512, out_features=CLASS_NUM)
	else:
		model.classifier._modules['6']= nn.Linear(4096, out_features=CLASS_NUM)  
if arg.useGPU_f:
	if torch.cuda.is_available():
		use_GPU = True
		model.cuda()
	else: 
		use_GPU = False
		model
		print("Error: NO GPU AVAILABLE, NOT USING GPU")
else:
	use_GPU = False
	print("Not using GPU")

if os.path.isfile(MODEL_PATH):
	model.load_state_dict(torch.load(MODEL_PATH))


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            wp = int(w*1.2)
            hp = int(h*1.2)
            xp = int(x-(wp-w)/2.)
            yp = int(y-(hp-h)/2.)
            if xp<0:
                xp = 0 
            if yp <0:
                yp = 0
            crop_img = img[yp:yp+hp, xp:xp+wp]
            #TODO Feed crop_img into pytorch network and label
            img_tensor = preprocess(img)
            img_tensor.unsqueeze_(0)
            img_variable = torch.Variable(img_tensor)
            #label should be converted to int here
            labelNum = 1
            label = classNames[labelNum]
            text = "Facial Expression: " +label
            
            if label == 'happy' or label == 'neutral' or label == 'surprise':
                color = (0,255,0)
            else:
                color = (0,0,255)
            cv2.putText(img,text,(x+10,y), FONT, 1,color,2,cv2.LINE_AA)
    cv2.imshow('live demo',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()