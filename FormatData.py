#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:09:16 2018

@author: keshavbachu
"""

import numpy as np

#load in the X and Y data

gameResults = np.load('/Users/keshavbachu/Documents/User Space/DASLAB Assignment/ctf_public-Release/gameResults.npy')
gameObservations = np.load('/Users/keshavbachu/Documents/User Space/DASLAB Assignment/ctf_public-Release/gameTrain.npy')


#using a list of 3 x3 arrays otherwise there is a problem with how numpy stores 4d arrays of different turn sizes
gameResults = gameResults[()]
gameObservations = gameObservations[()]