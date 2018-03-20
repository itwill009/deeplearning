# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:28:02 2018

@author: 마이마이
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:36:43 2018

@author: cdh66
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import tensorflow as tf
import numpy as np

from catdogload import catdog,next_batch
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

train_path = 'c:/data/dc/train/'

a,b = catdog(train_path)

trainX,testX,trainY,testY=train_test_split(a,b,test_size=0.2,random_state=6)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None,100,100,1])
Y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

###
## model
#
conv1=tf.layers.conv2d(inputs=X,filters=32,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu, kernel_initializer=he_init)
conv_batch_1 = tf.layers.batch_normalization(conv1,training=is_training)
pool1=tf.layers.max_pooling2d(inputs=conv_batch_1,pool_size=[2,2],padding='SAME',strides=2)

conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu, kernel_initializer=he_init)
conv_batch_2 = tf.layers.batch_normalization(conv2,training=is_training)
pool2=tf.layers.max_pooling2d(inputs=conv_batch_2,pool_size=[2,2],padding='SAME',strides=2)

conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu, kernel_initializer=he_init)
conv_batch_3 = tf.layers.batch_normalization(conv3,training=is_training)
pool3 = tf.layers.max_pooling2d(inputs=conv_batch_3, pool_size=[2, 2], padding='SAME', strides=2)

conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu, kernel_initializer=he_init)
conv_batch_4 = tf.layers.batch_normalization(conv4,training=is_training)
pool4 = tf.layers.max_pooling2d(inputs=conv_batch_4, pool_size=[2, 2], padding='SAME', strides=2)

#conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
#conv_batch_5 = tf.layers.batch_normalization(conv5,training=is_training)
#pool5 = tf.layers.max_pooling2d(inputs=conv_batch_5, pool_size=[2, 2], padding='SAME', strides=2)

#conv6 = tf.layers.conv2d(inputs=pool5, filters=256, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
#conv_batch_6 = tf.layers.batch_normalization(conv6,training=is_training)
#pool6 = tf.layers.max_pooling2d(inputs=conv_batch_6, pool_size=[2, 2], padding='SAME', strides=2)
#
#conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=[2,2],padding='VALID', activation=tf.nn.relu)
#conv_batch_7 = tf.layers.batch_normalization(conv7,training=is_training)
#pool7 = tf.layers.max_pooling2d(inputs=conv_batch_7, pool_size=[2, 2], padding='SAME', strides=2)

flat = tf.contrib.layers.flatten(pool4)
dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, kernel_initializer=he_init)
dense_batch_1 = tf.layers.batch_normalization(dense1,training=is_training)
d1=tf.nn.dropout(dense_batch_1,keep_prob)
dense2 = tf.layers.dense(d1, 128, activation=tf.nn.relu, kernel_initializer=he_init)
dense_batch_2 = tf.layers.batch_normalization(dense2,training=is_training)
d2=tf.nn.dropout(dense_batch_2,keep_prob)
model = tf.layers.dense(d2, 2, activation=None)

pred=tf.argmax(model,1)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))
#optimizer=tf.train.AdamOptimizer().minimize(cost)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer=tf.train.AdamOptimizer().minimize(cost)


is_correct=tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))  

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
#    saver=tf.train.Saver()
#    save_dir = './model/'
    
#    if not os.path.exists(save_dir):
#        os.makedirs(save_dir)
#        
#    save_path = os.path.join(save_dir, 'best_validation.ckpt')
 

 
    batch_size=100
    total_train_batch=int(trainX.shape[0]/batch_size)
    total_test_batch = int(testX.shape[0] / batch_size)

    for epoch in range(15):
        total_cost=0
        train_acc=0
        test_acc=0
        
        for i in range(total_train_batch):
          
            batch_xs, batch_ys=next_batch(trainX,trainY,i,batch_size)
            batch_xs=batch_xs.reshape(-1,100,100,1)
            
            _, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys,is_training:True,keep_prob:0.5})
#            print(batch_xs.shape)
            total_cost += cost_val
            
        for i in range(total_train_batch):
            
            batch_xs, batch_ys=next_batch(trainX,trainY,i,batch_size)
            batch_xs=batch_xs.reshape(-1,100,100,1)
            
            acc = sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys,is_training:True,keep_prob:0.5})
            train_acc += acc
            
        for i in range(total_test_batch):

            test_x, test_y = next_batch(testX,testY,i,batch_size)
            test_x=test_x.reshape(-1,100,100,1)
            val_acc = sess.run(accuracy, feed_dict={X:test_x,Y:test_y,is_training:False,keep_prob:1.0})
            test_acc += val_acc

            
        
        print('epoch:', epoch + 1, 'loss:', round(total_cost / total_train_batch,2), 'acc:', round(train_acc / total_train_batch,2), 'val_acc:', round(test_acc/total_test_batch,2))
        
        
#    predict = sess.run(pred, feed_dict={X:testX[0:100].reshape(-1,100,100,1),is_training:False,keep_prob:1.0})      
#    
#    true_label=['dog','cat']
#    
#    pred_list=[true_label[i] for i in predict]
#    true_list=[true_label[i] for i in np.argmax(testY[0:100],1)]
#    
#    
#   
#    
#    for j,k in zip(pred_list,true_list):
#        
#        if j == k:
#            print(j,k,'correct')
#        else:
#            print(j,k)
        

        
#    saver.save(sess,save_path)
            
   

 
#    print(save_path)






    