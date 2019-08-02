# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:50:15 2019

@author: Katerina
"""
#Thompson Sampling
#Probabilistic approach
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
###Bayesian Inference###
# ad i gets reward y from the Bernoulli distribution p(y|θi) ~ B(θi)
# θi is unknown but we set its uncertainty by assuming it has a uniform
# distribution p(θi) ~ U([0,1]), which is the prior distribution
# Bayes Rule: we approach θi by the posterior distribution
# p(θi|y) = p(y|θi)p(θi)/integral(p(y|θi)p(θi))dθi ~ p(y|θi) x p(θi),
# where p(y|θi) is the likelihood function and p(θi) the prior distribution
# we get p(θi|y) ~ β(number of successes + 1, number of failures +1)
# at each round n we take a random draw θi(n) from this posterior distribution p(θi|y) for each ad i
# for each round n we select the ad i that has the highest θi(n)
for n in range(0, N):
    ad = 0
    max_random_draw = 0
    for i in range(0, d):
        #for each ad i in the for i loop we are taking a random draw from the 
        #beta distribution of parameters number the times the ad got reward "1" +1
        #and number of times the ad got reward "0" +1
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1) #random draws taken from the beta distribution
        #each time we take a random draw from this distribution
        #we check to see if it is bigger than the max_random_draw and this is the case
        #the first time, since it starts at 0 
        #then we take another random draw from the beta distribution that correspond to the new ad i
        #if it is bigger, we select the new ad with the highest random draw
        if max_random_draw < random_beta:
            max_random_draw = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad] # checks the real value 0 or 1
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1  # adds 0 or 1 for the specific ad
    if reward == 0:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
#Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

