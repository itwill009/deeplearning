# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 00:40:13 2018

@author: 마이마이
"""

import csv

import os

import time

import cv2

import numpy as np

import re

import random





def image_load(path):

    start = time.time()

    file_list = os.listdir(path)

    file_name = []

    for i in file_list:

        a = int(re.sub('[^0-9]','',i))

        file_name.append(a)

        file_name.sort()

    data = []

    for i in file_name:

        file = path + str(i) + '.png'

        data.append(file)

    image = []

    for j in data:

        img = cv2.imread(j)

        #img = np.transpose(img) # 텐써 플로우 기본 모델은 채널이 맨 마지막에 와야하기 때문에

        image.append(img)        # 저번에 썼던 모델은 채널이 앞에 와야해서 썻다.

    image = np.array(image)



    end = time.time()

    print('image load time: %.2f' % float(end - start))

    return np.array(image)



def label_load(path):

    start = time.time()

    file = open(path)

    labeldata = csv.reader(file)

    labellist = []

    for i in labeldata:

        labellist.append(i)

    label = np.array(labellist)

    end = time.time()

    print('label load time: %.2f' % float(end - start))

    label = label.astype(int)

    label = np.eye(10)[label]

    label = np.squeeze(label,[-1,10])

    return label





def shuffle_batch(data_list, label):

    x= np.arange(len(data_list))

    random.shuffle(x)

    data_list2 = data_list[x]

    label2 = label[x]

    return data_list2, label2



def next_batch(data_list,label,idx,batch_size):



    batch1 = data_list[idx * batch_size:idx * batch_size + batch_size]



    label2 = label[idx * batch_size:idx * batch_size + batch_size]



    return batch1, label2