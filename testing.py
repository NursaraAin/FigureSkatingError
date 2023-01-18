# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:43:55 2023

@author: nursa
"""
import EdgeError_model2
import cv2
import numpy as np
import tensorflow as tf

model = EdgeError_model2.load_model()
cap = cv2.VideoCapture('C:/Users/nursa/OneDrive - Universiti Malaya/FigureSkatingError/E_test_correct (1).mp4')

frames=[]
classes = ['correct edge', 'unclear edge', 'wrong edge']
dim = (224,224)
ret = True
while ret:
    ret, img = cap.read()# read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        frames.append(img)
video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
score_tf = model.predict(tf.expand_dims(video,axis=0))
#print('The edge entry of this Flip/Lutz is ',classes[np.argmax(score_tf)],'(')

predict = dict.fromkeys(classes)
for i in range(len(classes)):
    predict[classes[i]]=score_tf[0][i]
    