#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:45:30 2018

@author: keshavbachu
"""
import numpy as np
import tensorflow as tf
import ModelTrainHelper as helper

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
def TrainModel(XTrain, rewards, actions, learning_rate = 0.0001, itterations = 2000, weights = None, biases = None, QWInput = None):
    qInput = rewards.reshape(rewards.shape[0])
    self_actions_input = actions.reshape(actions.shape[0])
    
    costs = []
    weights_store = []
    biases_store = []
    
    if(type(weights) != list and type(weights) != np.ndarray):
        trainSet, rewardSet, actionSet = helper.generatePlaceholders(XTrain, rewards, actions)
        targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        layer1, weightTemp, biasTemp = helper.conv_net(trainSet, XTrain.shape[3], 4, 32)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        layer2, weightTemp, biasTemp = helper.conv_net(layer1, 32, 4, 16)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        flattened = helper.flatten(layer2)
        
        fully_connected1, weightTemp, biasTemp = helper.fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 64)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        fully_connected2, weightTemp, biasTemp = helper.fc_layer(fully_connected1, fully_connected1.get_shape()[1:4].num_elements(), 5)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
    
    else:
        trainSet, rewardSet, actionSet = helper.generatePlaceholders(XTrain, rewards, actions)
        targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self_actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        layer1, weightTemp, biasTemp = helper.conv_net(trainSet, XTrain.shape[3], 4, 10,  weights[0], biases[0], layerNumber = 0)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        layer2, weightTemp, biasTemp = helper.conv_net(layer1, 10, 4, 8,  weights[1], biases[1], layerNumber = 1)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        flattened = helper.flatten(layer2)
        
        fully_connected1, weightTemp, biasTemp = helper.fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 64,  weights = weights[2], biases = biases[2], layerNumber = 2)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
        
        fully_connected2, weightTemp, biasTemp = helper.fc_layer(fully_connected1, fully_connected1.get_shape()[1:4].num_elements(), 4,  weights = weights[3], biases = biases[3], layerNumber = 3)
        weights_store.append(weightTemp)
        biases_store.append(biasTemp)
    
    
    #valueOutput = tf.nn.softmax_cross_entropy_with_logits(logits = fully_connected,labels=actionSet)
    #cost, prediction ,finalPrediction = helper.computeCost(fully_connected2, rewardSet, actionSet)
    cost, prediction ,finalPrediction, QW = helper.expReplayHelper(fully_connected2, targetQ, self_actions, QWIN = QWInput)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input}
    
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0
        for itter in range (itterations):
            _,temp_cost = sess.run([optimizer, cost], feed_dict=feed_dict)
            #check = sess.run([cost], feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions, targetQ: qInput, self_actions:self_actions_input})

            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
            
    
    #get the weights
        #calcTaken = tf.arg_max(fully_connected, 1)
        """
        predict = tf.arg_max(Qout, 1)
        randomProbability = tf.random_uniform([-1,], 0,1)
        randomDecision = tf.random_uniform([-1,], 0,4, tf.int32);
        #finalOutput = tf.cond(randomProbability < boundsFactor, lambda: tf.identity(randomDecision), lambda: tf.identity(predict))
        finalOutput = tf.where(randomProbability < 0.3, tf.cast(randomDecision, tf.int32), tf.cast(predict, tf.int32))
        """
        weights, biases, actionsTaken = sess.run([weights_store, biases_store, prediction], feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions})
        TrueVals = sess.run(finalPrediction, feed_dict={trainSet: XTrain, rewardSet: rewards, actionSet: actions})
        QWOut = sess.run(QW)
    return weights, biases, actionsTaken, TrueVals, QWOut

"""
Inputs:
    Xinput: X observation space to make predictions
    weights, biases: weights and biases needed to build the models
Outputs:
    actionsTaken: The action to take, determined from the model
"""
def makePredictions(Xinput, weights, biases, QW):
    #Construct the model and return predictions about the model
    trainSet, _, _ = helper.generatePlaceholders(Xinput, Xinput, Xinput)
    
    #construct the model to predict on
    layer1, _, _ = helper.conv_net(trainSet, Xinput.shape[3], 4, 10,  weights[0], biases[0], layerNumber = 0)
    layer2, _, _ = helper.conv_net(layer1, 10, 4, 8,  weights[1], biases[1], layerNumber = 1)
    
    flattened = helper.flatten(layer2)
    
    fully_connected1, _, _= helper.fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 64,  weights = weights[2], biases = biases[2], layerNumber = 2)
    fully_connected2, _, _ = helper.fc_layer(fully_connected1, fully_connected1.get_shape()[1:4].num_elements(), 4,  weights = weights[3], biases = biases[3], layerNumber = 3)

    QW = tf.Variable(tf.convert_to_tensor(QW, dtype = tf.float32))
    Qout = tf.matmul(fully_connected2, QW)
    predict = tf.arg_max(Qout, 1)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        calcTaken = tf.arg_max(fully_connected2, 1)
        actionsTaken = sess.run([predict], feed_dict={trainSet: Xinput})
    return actionsTaken