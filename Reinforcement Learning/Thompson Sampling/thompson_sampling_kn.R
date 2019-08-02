#Thompson Sampling

#Importing the dataset

dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = integer()
numbers_of_reward_1 = integer(d) # will create a vector of size d with 0s
numbers_of_reward_0 = integer(d)
total_reward = 0
#Bayesian Inference
#ad i gets reward y from the Bernoulli distribution p(y|θi) ~ B(θi)
# θi is unknown but we set its uncertainty by assuming it has a uniform
# distribution p(θi) ~ U([0,1]), which is the prior distribution
#Bayes Rule: we approach θi by the posterior distribution
#p(θi|y) = p(y|θi)p(θi)/integral(p(y|θi)p(θi))dθi ~ p(y|θi) x p(θi),
# where p(y|θi) is the likelihood function and p(θi) the prior distribution
#we get p(θi|y) ~ β(number of successes + 1, number of failures +1)
#at each round n we take a random draw θi(n) from this posterior distribution p(θi|y) for each ad i
#for each round n we select the ad i that has the highest θi(n)
for (n in 1:N) {
  ad = 0
  max_random = 0
  for(i in 1:d) {
    #for each ad i in the for i loop we are taking a random draw from the 
    #beta distribution of parameters number the times the ad got reward "1" +1
    #and number of times the ad got reward "0" +1
    random_beta = rbeta(n = 1, 
                        shape1 = numbers_of_reward_1[i]+1, 
                        shape2 = numbers_of_reward_0[i]+1) #random draws taken from the beta distribution
    #each time we take a random draw from this distribution
    #we check to see if it is bigger than the max_random_draw and this is the case
    #the first time, since it starts at 0 
    #then we take another random draw from the beta distribution that correspond to the new ad i
    #if it is bigger, we select the new ad with the highest random draw
    if (random_beta > max_random){
      max_random = random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1  # adds 0 or 1 for the specific ad
  }
  else {
    numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
  }
  total_reward = total_reward + reward
}


# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

