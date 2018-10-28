#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:45:30 2018

@author: keshavbachu
"""
import numpy as np
import tensorflow as tf
import ModelTrainHelper as helper
    
def TrainModel(XTrain, rewards, actions, learning_rate = 0.01, itterations = 500):
    qInput = rewards.reshape(rewards.shape[0])
    self_actions_input = actions.reshape(actions.shape[0])
    
    costs = []
    weights_store = []
    biases_store = []
    
    trainSet, rewardSet, actionSet = helper.generatePlaceholders(XTrain, rewards, actions)
    targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
    self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
    
    layer1, weightTemp, biasTemp = helper.conv_net(trainSet, XTrain.shape[3], 8, 10)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    flattened = helper.flatten(layer1)
    
    fully_connected, weightTemp, biasTemp = helper.fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 4)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    #valueOutput = tf.nn.softmax_cross_entropy_with_logits(logits = fully_connected,labels=actionSet)
    #cost = computeCost(fully_connected, rewardSet, actionSet)
    cost = helper.expReplayHelper(fully_connected, targetQ, self_actions)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    init = tf.global_variables_initializer()
    
    feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input}
    
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0
        for itter in range (itterations):
            _,temp_cost, check = sess.run([optimizer, cost, fully_connected], feed_dict=feed_dict)
            #check = sess.run([cost], feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input})

            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
            
    
    #get the weights
        calcTaken = tf.arg_max(fully_connected, 1)
        weights, biases, actionsTaken = sess.run([weights_store, biases_store, calcTaken], feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions})
    return weights, biases, actionsTaken
