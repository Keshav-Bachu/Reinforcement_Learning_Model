#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:38:49 2018

@author: keshavbachu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:45:30 2018

@author: keshavbachu
"""
import numpy as np
import tensorflow as tf
#from Reinforcement_Learning_Model import ModelTrainHelper as helper
import ModelTrainHelper as helper
Target = 100
Asynch = 10


"""
Input:
    Xtrain: Input values for the model
    rewards: Reward each action gets
    actions: Output taken by previous model
    learning_rate: Step size in gradient decent
    itterations: Number of itterations that need to be trained on
Output:
    weights, biases: weights and biases used/modified in the training of the unit
    actionsTaken: Predictions taken by the model

Trains the model and updates/makes weights/biases
"""
#change to use inherent TF stuff sometime
def TrainModel(XTrain, rewards, actions, learning_rate = 0.0001, itterations = 2000, weights = None, biases = None, QWInput = None):
    costs = []
    weights_store = []
    biases_store = []
    
    prevSetWeights = []
    prevSetBiases = []
    prevQW = []
    cost = 0
    

    trainSet, rewardSet, actionSet = helper.generatePlaceholders(XTrain, rewards, actions)
    targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
    self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
    
    layer1, weightTemp, biasTemp = helper.conv_net(trainSet, XTrain.shape[3], 4, 10)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    layer2, weightTemp, biasTemp = helper.conv_net(layer1, 10, 4, 8)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    flattened = helper.flatten(layer2)
    
    fully_connected1, weightTemp, biasTemp = helper.fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 64)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    fully_connected2, weightTemp, biasTemp = helper.fc_layer(fully_connected1, fully_connected1.get_shape()[1:4].num_elements(), 5)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    cost, prediction ,finalPrediction, QW = helper.expReplayHelper(fully_connected2, targetQ, self_actions, QWIN = QWInput, cost)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input}
    
    #store the weights for long update
    
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0
        for itter in range (itterations):
            #_,temp_cost = sess.run([optimizer, cost], feed_dict=feed_dict)
            #check = sess.run([cost], feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input})
            temp_cost = sess.run(cost, feed_dict = feed_dict)
            if(itter % Target == 0):
                prevSetWeights, prevSetBiases, prevQW = sess.run([weights_store, biases_store, QW], feed_dict = feed_dict)
            
            if(itter % Asynch == 0):
                _ = sess.run(optimizer, feed_dict = feed_dict)
                cost = 0
            
            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
            
    
   
    return None
