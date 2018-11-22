#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:59:03 2018

@author: keshavbachu
"""


"""
finalLayer: final layer of the NN before calculations
h_size: "The size of the final convolutional layer before splitting it into Advantage and Value streams"
actions: Number of available actions 
"""

def expReplayHelper(finalLayer, targetQ, self_actions, h_size = 5, actions = 5, QWIN = None, itterationNum, loss = 0):
    #streamQ = tf.layers.flatten(finalLayer)
    streamQ = finalLayer
    xavier_init = tf.contrib.layers.xavier_initializer()
    
    if(type(QWIN) != np.ndarray):
        QW  = tf.Variable(xavier_init([h_size, actions]))
    else:
        #QW = tf.get_variable(name = "QW", initializer = QWIN)
        QW = tf.Variable(tf.convert_to_tensor(QWIN, dtype = tf.float32))
    Qout = tf.matmul(streamQ, QW)
    
    predict = tf.arg_max(Qout, 1)
    randomProbability = tf.random_uniform(tf.shape(predict), 0,1)
    randomDecision = tf.random_uniform(tf.shape(predict), 0,5, tf.int32)
    #finalOutput = tf.cond(randomProbability < boundsFactor, lambda: tf.identity(randomDecision), lambda: tf.identity(predict))
    finalOutput = tf.where(randomProbability < 0.3, tf.cast(randomDecision, tf.int32), tf.cast(predict, tf.int32))
 
    #targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
    #self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
    actions_onehot = tf.one_hot(self_actions, actions,dtype=tf.float32)

    Q = tf.reduce_sum(tf.multiply(Qout, actions_onehot), axis=1)

    td_error = tf.square(targetQ - Q)
    loss += tf.reduce_mean(td_error)
    
    return loss, predict, finalOutput, QW