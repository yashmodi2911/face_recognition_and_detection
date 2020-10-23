# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:32:58 2020

@author: hp
"""

import os
import numpy as np
from PIL import Image
import cv2


face_cascade=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
Base_dir=os.path.dirname(os.path.abspath('images'))
image_dir=os.path.join(Base_dir,"images")
recognizer=cv2.face.LBPHFaceRecognizer_create()

label_ids={}
current_id=0
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            path=os.path.join(root,file)#path of image
            label=os.path.basename(root).replace(" ","-").lower()#in which direct folder it's installed
            print(label,path)
            
            if label in label_ids:
                pass
            else:
                label_ids[label]=current_id;
                current_id+=1;
                
            id_=label_ids[label]    
            pil_image=Image.open(path).convert("L")#image into grayscale
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(final_image,"uint8")#image into numpy array
           
            faces=face_cascade.detectMultiScale(image_array)
            
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                

print(x_train)
print(y_labels)
import pickle

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
    
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")