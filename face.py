# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:16:30 2020

@author: Yash Modi
"""

import numpy as np
import cv2

import pickle
labels={}

with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}#reverse the key & value
recognizer=cv2.face.LBPHFaceRecognizer_create()
eye_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer.read("trainner.yml")

cap=cv2.VideoCapture(0)
while(True):
    check,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            text_color=(255,255,255)
            stroke=1
            cv2.putText(frame,name, (x,y) ,font,1,text_color,stroke,cv2.LINE_AA)
        
        #print (x,y,w,h)
        color=(255,0,0)
        stroke=2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('Capturing',frame)
    
    key=cv2.waitKey(20)
    if(key==ord('q')):
        break;
    
cap.release()
cv2.destroyAllWindows()

