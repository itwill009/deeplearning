# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 01:09:52 2018

@author: 마이마이
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sqlite3

df = pd.read_csv('e:/data/creditcard.csv')

# 데이터 정규화
df_norm = (df - df.min() ) / (df.max() - df.min() )

fraud=df_norm[df_norm.Class==1.0] 
normal=df_norm[df_norm.Class==0.0] 

# normal data로만 학습할 것이기때문에 train 80 test 20 으로 나눈다
# 학습은 normal data 80% 로만 한다
train=normal.sample(frac=0.8)
trainX=train.iloc[:,1:29].as_matrix()

# 테스트는 20% 의 normal data 와 fraud data
test=normal.loc[~normal.index.isin(train.index)]
test=test.sample(frac=1)
testX=test.iloc[:,1:29].as_matrix()

fraudX=fraud.iloc[:,1:29].as_matrix()

normalX=normal.iloc[:,1:29].as_matrix()


print('Total train :', trainX.shape[0])
print()

# 그래프 생성

X=tf.placeholder(tf.float32,[None,28])

# encoder
dense1=tf.layers.dense(inputs=X,units=20,activation=tf.nn.sigmoid)
dense2=tf.layers.dense(inputs=dense1,units=14,activation=tf.nn.sigmoid)
encoder = tf.layers.dense(inputs=dense2,units=10,activation=tf.nn.sigmoid)

# decoder
dense4 = tf.layers.dense(inputs=encoder,units=14,activation=tf.nn.sigmoid)
dense5 = tf.layers.dense(inputs=dense4,units=20,activation=tf.nn.sigmoid)
decoder = tf.layers.dense(inputs=dense5,units=28,activation=tf.nn.sigmoid)

# optimizer
cost = tf.reduce_sum(tf.square(tf.subtract(decoder,X)))
optimizer = tf.train.AdamOptimizer().minimize(cost)


def next_batch(data,i,batch):
    
    return data[i*batch:i*batch+batch]

print('training')

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    batch_size=500
    
        
    total_train_batch=int(trainX.shape[0]/batch_size)

    for epoch in range(3):
        
        total_cost=0
      
        
        for i in range(total_train_batch):
          
            batch_xs = next_batch(trainX,i,batch_size)
                        
            _, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs})

            total_cost += cost_val
        
        
        
        print('epoch:', epoch + 1, 'cost:', round(total_cost / total_train_batch,2))
       
    print('train complete!')
    print() 
    print('testing..')
    
    
    
#    limit_list=[0.001,0.01,0.05,0.07,0.1,0.5]
    
#    for limit in limit_list:
        
    fraud_count = 0
    normal_count = 0

    print('임계값 :', 0.07)

    a_list=[]
    b_list=[]
     
    for i in range(len(fraud)):
        
        a=sess.run(cost,feed_dict={X:fraudX[i].reshape(1,28)})
        
        if a >= 0.07:
        
            fraud_count +=1
            
            a_list.append(fraudX[i])
    
    for i in range(len(testX[0].shape)):
        
        b=sess.run(cost,feed_dict={X:testX[i].reshape(1,28)})
            
        if b >= 0.07:
            
            normal_count +=1
                
#            b_list.append(b)
                
          
            
    print('fraud :', fraud_count,'/',len(fraud), 'normal :', normal_count,'/',len(testX))
   
    maybefraud = pd.DataFrame(a_list)
    con = sqlite3.connect("company.db")
    maybefraud.to_sql('maybefraud',con)
    
    
    
#    bins = 20
#    
#    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
#    
#    ax1.hist(a, bins = bins)
#    ax1.set_title('Fraud')
#
#    ax2.hist(b, bins = bins)
#    ax2.set_title('Normal')
#    
#    plt.xlabel('cost')
#    plt.ylabel('Number of transaction')
#    
#    plt.show()
