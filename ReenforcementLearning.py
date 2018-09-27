#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:45:30 2018

@author: keshavbachu
"""
import numpy as np
import tensorflow as tf

def decaying_Reward(rewardSet):
    decayedReward = np.zeros(rewardSet.shape())
    gamma = 0.99
    
    for i in range (rewardSet.shape[0]):
        continuousAdd = 0
        for j in range(rewardSet.shape[1]):
            continuousAdd = continuousAdd * gamma + rewardSet[i][j]
            decayedReward[i][j] = continuousAdd

def generatePlaceholders(trainX, trainReward, trainAction):
    dim1X = trainX.shape[1]
    dim2X = trainX.shape[2]
    dim3X = trainX.shape[3]
    
    rewardDim1 = trainReward.shape[1]
    
    actionDim1 = trainAction.shape[1]
    
    Xtrain = tf.placeholder(shape = [None, dim1X, dim2X, dim3X], dtype=tf.float32)
    rewardTrain = tf.placeholder(shape = [None, rewardDim1], dtype=tf.float32)
    actionTrain = tf.placeholder(shape = [None, actionDim1], dtype=tf.float32)
    
    return Xtrain, rewardTrain, actionTrain
    
def conv_net(input_data, num_input_channels, filter_shape, num_filters):
    #weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    conv_filt_shape = [filter_shape,filter_shape, num_input_channels, num_filters]
    
    weights = create_weights(conv_filt_shape)
    bias = create_biases(num_filters)
    
    out_layer = tf.nn.conv2d(input=input_data, filter= weights, strides= [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.max_pool(value=out_layer, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
    out_layer = tf.nn.relu(out_layer)
    
    return out_layer, weights, bias
    
def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 

def fc_layer(input,num_inputs,num_outputs, use_relu = False):
    weights = create_weights(shape=[num_inputs, num_outputs])
    
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if(use_relu == True):
        layer = tf.nn.relu(layer)
        
    return layer, weights, biases


def ReplayNetworkModel(Xplaceholders, Yplaceholders):
    conv_layer1, weightTemp, biasTemp = conv_net(x, x.shape[3], 8, 10)
    
    
    
def TrainModel(XTrain, Y):
    