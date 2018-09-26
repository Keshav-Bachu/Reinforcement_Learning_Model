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

def generatePlaceholders(trainX, trainY):
    
def ReplayNetwork():
    
    
    
def TrainModel(XTrain, Y):
    