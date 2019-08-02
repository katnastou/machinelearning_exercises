# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 22:07:00 2019

@author: Infinity
"""
#UCB
#deterministic Approach
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing Upper Confidence Bound (UCB)
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        #during the 10 first rounds we do not have enough info
        #so we need to set the initial conditions
        #whether reward equals 0 or 1 since we haven't selected them yet
        #so for the 10 first rounds we will simply select the 10 ads 
        #with no strategy, that will give us some info about the 10 ads
        #since the number_of_selections will be 1 for all ads
        #and we will also get info about the rewards
        if (numbers_of_selections[i] > 0): # the strategy will apply after the first 10 rounds
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i]) # index starts at 0, log(0) undefined
            upper_bound = average_reward + delta_i

        else:
            upper_bound = 1e400
        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad] # checks the real value 0 or 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  # adds 0 or 1 for the specific ad
    total_reward = total_reward + reward
    
#Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

