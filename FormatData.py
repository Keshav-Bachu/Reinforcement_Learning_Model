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
        return lastLocX - 1, lastLocY
   
    elif(lastLocY != 0 and specificGame[lastLocY - 1][lastLocX] == valueFind):
        return lastLocX, lastLocY - 1
    
    elif(lastLocX != specificGame.shape[1] - 1 and specificGame[lastLocY][lastLocX + 1] == valueFind):
        return lastLocX + 1, lastLocY
    
    elif(lastLocY != specificGame.shape[0] - 1 and specificGame[lastLocY + 1][lastLocX] == valueFind):
        return lastLocX, lastLocY + 1
    
    return lastLocX, lastLocY

def getObservations(specificGame, observationSize, lastLocX, lastLocY):
    leftSide = max(lastLocX - observationSize, 0)
    rightSide = min(lastLocX + observationSize, specificGame.shape[1] - 1)
    
    topSide = max(lastLocY - observationSize, 0)
    bottomSide = min(lastLocY + observationSize, specificGame.shape[0] - 1)
    
    return specificGame[topSide:bottomSide + 1 ,leftSide:rightSide + 1]

def addPadding(observationSpace, objectpad, observationSpace):
    
    
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

#observe off of one 
for game in gameObservations:
    #find the initial location of a piece to observe
    for turn in game:
        if firstTurn:
            firstTurn = False
            locX, locY = initialLocation(turn, 4)
        else:
            locX, locY = nextLocation(turn, 4, locX, locY)
            
        observationSpace = getObservations(turn, 2, locX, locY)
        print(observationSpace, '\n')