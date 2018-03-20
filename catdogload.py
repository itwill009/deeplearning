# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:38:58 2018

@author: 마이마이
"""

import numpy as np
import cv2
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import random

def catdog(path):

    start_time=time.time()
    
    random.seed(2)
    
    a=os.listdir(path)
    
    b=[path+i for i in a]
    
    random.shuffle(b)
    
    image=[]
    label=[]
    
    for i in b:
        
        img=cv2.imread(i,0)
        img=cv2.resize(img,(100,100))
        image.append(img)
        
        
        if 'dog' in i:
            label.append(1)
        else:
            label.append(2)
    
       
    image=np.array(image)
    image=np.clip(image/255.0,0.0,1.0)
    
    end_time=time.time()
    print(end_time-start_time)
       
    df = pd.Series(label)
    #print(df)
    
    label=pd.get_dummies(df).as_matrix()
    
    #print(label)
    
    #plt.imshow(image[100])
    #true_label=['dog','cat']
    #print(true_label[np.where(label[100]==1)[0][0]])
    
    return image,label

def next_batch(image,label,i,batch):
    
    return image[i*batch:i*batch+batch], label[i*batch:i*batch+batch]







