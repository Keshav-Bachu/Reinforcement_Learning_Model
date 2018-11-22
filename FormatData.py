#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:09:16 2018

@author: keshavbachu
"""

import numpy as np
import ReenforcementLearning as  REL


"""
Inputs:
    specificGame: The game that is being looked at
    valueFind: The value we are observing upon
Outputs:
    i, j: x and y coordiantes of the value being looked upon

Searches an individual game for a value and return the coordinates of the first instance of the value
"""
def initialLocation(specificGame, valueFind):
    for i in range(specificGame.shape[1]): #cols
        for j in range(specificGame.shape[0]): #rows
            if(specificGame[j][i] == valueFind):
                return i, j
            
    return -1, -1

"""
Inputs:
    specificGame: The game that is being looked at
    valueFind: The value we are observing upon
    lastLocX: Previous X value used, to make sure same unit is observed
    lastLocY: Previous Y value used, to make sure same unit is observed
Outputs:
    Moved location of object and action it took

Determine the next location of the unit 
"""
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

"""
Input:
    specificGame: The game that is being looked at 
    observationSize: The observation space that the unit can look at
    lastLocX: Previous X value used, to make sure same unit is observed
    lastLocY: Previous Y value used, to make sure same unit is observed
Outputs:
    Returns specific game frame
    
Takes a specific frame interval for the observation windows
"""
def getObservations(specificGame, observationSize, lastLocX, lastLocY):
    leftSide = max(lastLocX - observationSize, 0)
    rightSide = min(lastLocX + observationSize, specificGame.shape[1] - 1)
    topSide = max(lastLocY - observationSize, 0)
    bottomSide = min(lastLocY + observationSize, specificGame.shape[0] - 1)
    
    return specificGame[topSide:bottomSide + 1 ,leftSide:rightSide + 1]

"""
Inputs:
    observation: The observations outputted from the function above
    objectpad: The value that can be padded for the observation space
    observationSpace: The size of the observation window
    objectLook: Object that we need to look for in observation
Output:
    PAdded array of the observations so the size is all the same
"""
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

def removeDeadTurns(actionAll, games, gameResults):
    maxLength = len(gameResults)
    i = 0
    while (i < maxLength):
        if(actionAll[i] == 0):
            actionAll = np.delete(actionAll, i, 0)
            games = np.delete(games, i, 0)
            gameResults = np.delete(gameResults, i, 0)
            i -= 1
            maxLength -= 1
        i += 1
            
            
    return actionAll, games, gameResults

def deadTurnsWrapper(actionAll, games, gameResults):
    for i in range(len(actionAll)):
        actionAll[i], games[i], gameResults[i] = removeDeadTurns(actionAll[i], games[i], gameResults[i])
        
    return actionAll, games, gameResults
        
 
def oneTurnObservation():
    #Start of the main program
    #load in the X and Y data
    
    gameResults = np.load('Data Folder/gameResults.npy')
    gameObservations = np.load('Data Folder/gameTrain.npy')
    
    
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
        firstTurn = True
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
    
    games = games.reshape(games.shape[0] * games.shape[1], games.shape[2], games.shape[3], 1)
    gameResults = gameResults.reshape(gameResults.shape[0] * gameResults.shape[1], 1)
    actionAll = actionAll.reshape(actionAll.shape[0] * actionAll.shape[1], 1)
    
    actionAll, games, gameResults = removeDeadTurns(actionAll, games, gameResults)
    
    #weights, biases, actionsTaken, actionsExploration, qouts = REL.TrainModel(games, gameResults, actionAll, itterations = 30000)
    #tf.reset_default_graph()
    #REL.makePredictions(games, weights, biases)
    
    #weights, biases, actionsTaken, actionsExploration = REL.TrainModel(games, gameResults, actionAll, itterations = 1000, weights = weights, biases = biases)
    #REL.TrainModel(games, gameResults, actionAll, itterations = 1000, QWInput = qout, weights = weights, biases = biases)

def intoExperience(obsLength, turns, rewards, actions):
    finalGames = []
    finalRewards = []
    finalActions = []
    totalGames = []
    for j in range(len(turns)):
        i = turns[j]
        if(len(totalGames) < obsLength):
            totalGames.append(i)
        else:
            finalGames.append(np.asanyarray(totalGames))
            del totalGames[0]
            totalGames.append(i)
            finalRewards.append(rewards[i])
            finalActions.append(actions[i])
    
    finalGames.append(np.asanyarray(totalGames))
    finalRewards.append(rewards[rewards.shape[0] - 1])
    finalActions.append(actions[actions.shape[0] - 1])
    
    finalGames = np.asanyarray(finalGames)
    finalRewards = np.asanyarray(finalRewards)
    finalActions = np.asanyarray(finalActions)
    
    return finalGames

def intoExperienceWrapper(obsLength, turns, rewards, actions):
    for i in range(len(turns)):
        intoExperience(obsLength, turns[i], rewards[i], actions[i])
            

def experienceReplayPreprocess(turnObsLength):
    #Start of the main program
    #load in the X and Y data
    
    gameResults = np.load('Data Folder/gameResults.npy')
    gameObservations = np.load('Data Folder/gameTrain.npy')
    
    
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
        action  = np.asanyarray(action)
        actionAll.append(action)
        firstTurn = True
    singleReward = np.zeros(gameResults.shape)
    
    
    for gameresult in range (0, gameResults.shape[0]):
        lastScore = 0
        for value in range(0, gameResults.shape[1]):
            #print(gameresult, " ", value)
            if(gameResults[gameresult][value] != lastScore):
                singleReward[gameresult][value] = gameResults[gameresult][value] - lastScore
                lastScore = gameResults[gameresult][value]
    
    #games = games.reshape(games.shape[0] * games.shape[1], games.shape[2], games.shape[3], 1)
    #gameResults = gameResults.reshape(gameResults.shape[0] * gameResults.shape[1], 1)
    resultsList = []
    for i in range(gameResults.shape[0]):
        resultsList.append(gameResults[i])
    
    #current turn corresponds to next action and reward so shift by 1
    for i in range(len(actionAll)):
        temp = actionAll[i][0]
        actionAll[i] = np.delete(actionAll[i], 0, 0)
        actionAll[i] = np.append(actionAll[i],temp)
        
        resultsList[i] = np.delete(resultsList[i], 0, 0)
        resultsList[i] = np.append(resultsList[i], resultsList[i][resultsList[i].shape[0] - 1])
    
    #have games with no dead turns, now need to make each of actionAll into the specific blocks
    actionAll, games, resultsList = deadTurnsWrapper(actionAll, games, resultsList)
    intoExperienceWrapper(turnObsLength, games, resultsList, actionAll)
    
experienceReplayPreprocess(9)