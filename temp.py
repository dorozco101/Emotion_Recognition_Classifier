#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:38:01 2018

@author: david
"""

import os
import numpy as np
import math

path = './data/train'
testPath = './data/test'
for dataset in os.listdir(path):
    for emotion in os.listdir(path+'/'+dataset):
        pictures =  os.listdir(path+'/'+dataset+'/'+emotion)
        indexs  = np.arange(len(pictures))
        np.random.shuffle(indexs)
        if not os.path.exists(testPath+'/'+emotion):
            os.mkdir(testPath+'/'+emotion)
        for i in range(math.floor(len(pictures)*.10)):
            pic = path+'/'+dataset+'/'+emotion+'/'+pictures[indexs[i]]
            os.rename(pic,testPath+'/'+emotion+'/'+pictures[indexs[i]])