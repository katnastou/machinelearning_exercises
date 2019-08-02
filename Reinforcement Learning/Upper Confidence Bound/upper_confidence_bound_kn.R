#Upper Confidence Bound

#Importing the dataset

dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10
ads_selected = integer()
numbers_of_selections = integer(d) # will create a vector of size d with 0s
sums_of_rewards = integer(d)
total_reward = 0

for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for(i in 1:d) {
    #during the 10 first rounds we do not have enough info
    #so we need to set the initial conditions
    #whether reward equals 0 or 1 since we haven't selected them yet
    #so for the 10 first rounds we will simply select the 10 ads 
    #with no strategy, that will give us some info about the 10 ads
    #since the number_of_selections will be 1 for all ads
    #and we will also get info about the rewards
    if (numbers_of_selections[i] > 0){ # the strategy will apply after the first 10 rounds
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    }
    else { # very large value for the first 10 rounds --> make sure you get all the ads
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad]+1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward #if it is selected it will be +1
  total_reward = total_reward + reward
}


# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

