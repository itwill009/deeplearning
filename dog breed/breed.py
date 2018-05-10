import pandas as pd
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#img2arr

def doggy():

    labels = pd.read_csv('c:/data/final/labels.csv')
    
#    labels=labels[0:1000]
    
    train_path = 'c:/data/final/train/'
    
    labels['image_path'] = labels.apply( lambda x: ( train_path + x["id"] + ".jpg" ), axis=1 )
    
    start_time=time.time()

    image=[]
              
    for i in labels['image_path']:
        
        img=cv2.imread(i,0)
        img=cv2.resize(img,(100,100))
        image.append(img)
        
    end_time=time.time()
                
    print(end_time-start_time)
    
    image=np.array(image)
    image=np.clip(image/255.0,0.0,1.0)
    
    breed=sorted(list(labels['breed'].value_counts().index))
    one_hot=pd.get_dummies(labels['breed']).as_matrix()
#    print(a[0])
#    print(breed[np.where(a[0]==1)[0][0]])
    
    return image,one_hot, breed

def next_batch(data,label,i,batch):
    
    batchx=data[i*batch:i*batch+batch]
    batchy=label[i*batch:i*batch+batch]
    
    
    return batchx,batchy



#
#a,b,c=doggy()
#
#print(b[3])
#print(c)
#print(c[np.where(b[1]==1)[0][0]])
#plt.imshow(a[1])


#trainX,testX,trainY,testY=train_test_split(a,b,test_size=0.2,random_state=6)
#print(trainX.shape)
#plt.imshow(trainX[1])
#print(c[np.where(trainY[1]==1)[0][0]])
#print(c[np.where(testY[1]==1)[0][0]])     
#plt.imshow(testX[1])
#            
