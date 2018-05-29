# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt

def results(loss_file, acc_file, net)


#loss_file = 'AlexNetTrainLossEmotions.npy'
#acc_file = 'AlexNetTrainAccuracyEmotions.npy'
#net = 'AlexNet'

loss = np.load(loss_file)
accuracy = 100.0*np.load(acc_file)

plt.plot(loss)
plt.title('Training Loss vs Epoch, ' + net)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(accuracy)
plt.title('Training Accuracy vs Epoch, ' + net)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()