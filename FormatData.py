#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:09:16 2018

@author: keshavbachu
"""

import numpy as np

#numpy 20 x 20 input
def initialLocation(specificGame, valueFind):
    for i in range(specificGame.shape[1]): #cols
        for j in range(specificGame.shape[0]): #rows
            if(specificGame[j][i] == valueFind):
                return i, j
            
    return -1, -1

#check all 4 dir for the object next position
def nextLocation(specificGame, valueFind, lastLocX, lastLocY):
    #check left
    if(lastLocX != 0 and specificGame[lastLocY][lastLocX - 1] == valueFind):
        return lastLocX - 1, lastLocY, 1
   
    elif(lastLocY != 0 and specificGame[lastLocY - 1][lastLocX] == valueFind):
        return lastLocX, lastLocY - 1, 2
    
    elif(lastLocX != specificGame.shape[1] - 1 and specificGame[lastLocY][lastLocX + 1] == valueFind):
        return lastLocX + 1, lastLocY, 3
    
    elif(lastLocY != specificGame.shape[0] - 1 and specificGame[lastLocY + 1][lastLocX] == valueFind):
        return lastLocX, lastLocY + 1, 4
    
    return lastLocX, lastLocY, 0

def getObservations(specificGame, observationSize, lastLocX, lastLocY):
    leftSide = max(lastLocX - observationSize, 0)
    rightSide = min(lastLocX + observationSize, specificGame.shape[1] - 1)
    
    topSide = max(lastLocY - observationSize, 0)
    bottomSide = min(lastLocY + observationSize, specificGame.shape[0] - 1)
    
    return specificGame[topSide:bottomSide + 1 ,leftSide:rightSide + 1]

def addPadding(observation, objectpad, observationSpace, objectLook):
    
    #horz padding
    if(observation.shape[1] != observationSpace):
        horzPadding = np.zeros((observation.shape[0], 1)) + objectpad
        
        #check if the values need to be added to the left or right
        addLeft = False
        for i in range(observation.shape[0]):
            for j in range(observationSpace):
                if(observation[i][j] == objectLook):
                    addLeft = True
            
        while(observation.shape[1] < 2 * observationSpace + 1):
            if(addLeft):
                observation = np.append(horzPadding, observation ,axis = 1)
            else:
                observation = np.append(observation, horzPadding ,axis = 1)
        
        
    if(observation.shape[0] != observationSpace):
        vertPadding = np.zeros((1, observation.shape[1])) + objectpad
        
        addTop = False
        for i in range(observation.shape[1]):
            for j in range(observationSpace):
                if(observation[j][i] == objectLook):
                    addTop = True
                    break
                    
        while(observation.shape[0] < 2 * observationSpace + 1):
            if(addTop):
                observation = np.append(vertPadding, observation, axis = 0)
            else:
                observation = np.append(observation, vertPadding ,axis = 0)
            
            
    return observation
        
        
    
    
#load in the X and Y data

gameResults = np.load('/Users/keshavbachu/Documents/User Space/DASLAB Assignment/ctf_public-Release/gameResults.npy')
gameObservations = np.load('/Users/keshavbachu/Documents/User Space/DASLAB Assignment/ctf_public-Release/gameTrain.npy')


#gameResults: [# Examples, turns, observation.shape[0], observation.shape[1]]
#Storage of all the observed values within the system
gameResults = gameResults[()]

#gameResults: [# Examples, turn limit]
#The results of the game represented by a score
gameObservations = gameObservations[()]


locX = -1;
locY = -1;
observationSpace = None
firstTurn = True

turns = []
games = []
actionAll = []

#observe off of one 
for game in gameObservations:
    turns = []
    action = []
    actionTaken = -1
    #find the initial location of a piece to observe
    for turn in game:
        if firstTurn:
            firstTurn = False
            locX, locY = initialLocation(turn, 4)
            actionTaken = 0
        else:
            locX, locY, actionTaken = nextLocation(turn, 4, locX, locY)
            
        observationSpace = getObservations(turn, 2, locX, locY)
        observationSpace = addPadding(observationSpace, objectpad=-1, observationSpace = 2, objectLook = 4)
        #print(observationSpace, '\n')
        turns.append(observationSpace)
        action.append(actionTaken)
    turns = np.asanyarray(turns)
    games.append(turns)
    actionAll.append(action)
games = np.asanyarray(games)
singleReward = np.zeros(gameResults.shape)
actionAll = np.asanyarray(actionAll)

for gameresult in range (0, gameResults.shape[0]):
    lastScore = 0
    for value in range(0, gameResults.shape[1]):
        #print(gameresult, " ", value)
        if(gameResults[gameresult][value] != lastScore):
            singleReward[gameresult][value] = gameResults[gameresult][value] - lastScore
            lastScore = gameResults[gameresult][value]
            
        