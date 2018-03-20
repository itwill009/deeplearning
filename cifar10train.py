# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 00:41:41 2018

@author: 마이마이
"""

import tensorflow as tf
import numpy as np
from cifarload import *

train_image = 'c:/data/cifar10/train/'
train_label = 'c:/data/cifar10/train_label.csv'
test_image = 'c:/data/cifar10//test/'
test_label = 'c:/data/cifar10/test_label.csv'


print("LOADING DATA")


trainX = image_load(train_image)
print(trainX.shape) # (50000, 3, 32, 32)
trainY = label_load(train_label)
print(trainY.shape) # (50000, 10)

testX = image_load(test_image)

print(testX.shape) # (10000, 3, 32, 32)

testY = label_load(test_label)

print(testY.shape) # (10000, 10)

testX, testY = shuffle_batch(testX, testY)



tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None,32,32,3])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)
###
## model
#
conv1=tf.layers.conv2d(inputs=X,filters=32,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu)
conv_batch_1 = tf.layers.batch_normalization(conv1,training=is_training)
pool1=tf.layers.max_pooling2d(inputs=conv_batch_1,pool_size=[2,2],padding='SAME',strides=2)

conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu)
conv_batch_2 = tf.layers.batch_normalization(conv2,training=is_training)
pool2=tf.layers.max_pooling2d(inputs=conv_batch_2,pool_size=[2,2],padding='SAME',strides=2)

conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
conv_batch_3 = tf.layers.batch_normalization(conv3,training=is_training)
pool3 = tf.layers.max_pooling2d(inputs=conv_batch_3, pool_size=[2, 2], padding='SAME', strides=2)
#
conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
conv_batch_4 = tf.layers.batch_normalization(conv4,training=is_training)
pool4 = tf.layers.max_pooling2d(inputs=conv_batch_4, pool_size=[2, 2], padding='SAME', strides=2)

#conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[1, 1], padding='SAME', activation=tf.nn.relu)
#conv_batch_5 = tf.layers.batch_normalization(conv5,training=is_training)
#pool5 = tf.layers.max_pooling2d(inputs=conv_batch_5, pool_size=[2, 2], padding='SAME', strides=2)

# conv6 = tf.layers.conv2d(inputs=pool5, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
# pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], padding='SAME', strides=2)

flat = tf.contrib.layers.flatten(pool4)
dense1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
dense_batch_1 = tf.layers.batch_normalization(dense1,training=is_training)
dense2 = tf.layers.dense(dense_batch_1, 128, activation=tf.nn.relu)
dense_batch_2 = tf.layers.batch_normalization(dense2,training=is_training)
model = tf.layers.dense(dense_batch_2, 10, activation=None)


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

    for epoch in range(20):
        
        total_cost=0
        train_acc=0
        val_acc=0
        
        for i in range(total_train_batch):
          
            batch_xs, batch_ys = next_batch(trainX,trainY,i,batch_size)
            
            _, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys,is_training:True})
            
            total_cost += cost_val
            
        for i in range(total_train_batch):
            
            batch_xs, batch_ys = next_batch(trainX,trainY,i,batch_size)
            
            acc = sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys,is_training:True})

            train_acc += acc

        for i in range(total_test_batch):

            test_x, test_y = next_batch(testX,testY,i,batch_size)

            test_acc = sess.run(accuracy,feed_dict={X:test_x,Y:test_y,is_training:False}) 

            val_acc += test_acc


        print('epoch:', epoch + 1, 'loss:', round(total_cost/total_train_batch,2), 'acc:', round(train_acc/total_train_batch,2), 'val_acc:', round(val_acc/total_test_batch,2))

