#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:25:51 2018

@author: keshavbachu
"""
#import FormatData
#import ReenforcementLearning as  REL
import numpy as np
import tensorflow as tf
from Reinforcement_Learning_Model import ModelCalls as Model
from Reinforcement_Learning_Model import FormatData
from Reinforcement_Learning_Model import ReenforcementLearning as  REL
from Reinforcement_Learning_Model import ModelTrainHelper as helper




def generateAllLocations(fieldView, unitObserve):
    AllLocations = []
    for i in range(fieldView.shape[1]): #cols
        for j in range(fieldView.shape[0]): #rows
            if(fieldView[j][i] == unitObserve):
                AllLocations.append([i,j])
    
    return AllLocations

#can use agent locations to check generate observational units                
def generateOutputs(fieldView, unitObserve, weights, biases, QW):
    AllLocations = []
    AllObservations = []
    AllLocations = generateAllLocations(fieldView, unitObserve)
    
    for i in AllLocations:
        temp = FormatData.getObservations(fieldView, 2, i[0], i[1])
        temp = FormatData.addPadding(temp, objectpad=-1, observationSpace = 2, objectLook = 4)
        
        AllObservations.append(temp)
    AllObservations = np.asanyarray(AllObservations)
    AllObservations = AllObservations.reshape(AllObservations.shape[0], AllObservations.shape[1], AllObservations.shape[2], 1)
    predictions = REL.makePredictions(AllObservations, weights, biases, QW)

    return predictions[0]

def generateFromLocation(fieldView, unitLocation, weights, biases, QW):
    AllLocations = unitLocation
    #AllLocations.append(unitLocation)
    AllObservations = []
    #AllLocations = generateAllLocations(fieldView, unitObserve)
    for i in AllLocations:
        temp = FormatData.getObservations(fieldView, 2, i[0], i[1])
        #print(unitLocation)
        #print(fieldView)
        #print(temp)
        temp = FormatData.addPadding(temp, objectpad=-1, observationSpace = 2, objectLook = 4)
        #print(temp)
        
        
        AllObservations.append(temp)
    AllObservations = np.asanyarray(AllObservations)
    AllObservations = AllObservations.reshape(AllObservations.shape[0], AllObservations.shape[1], AllObservations.shape[2], 1)
    predictions = REL.makePredictions(AllObservations, weights, biases, QW)

    return predictions[0]


#pred = generateOutputs(gameObservations, 4, weights, biases, QW)
#generateFromLocation(gameObservations, location, weights, biases, QW)