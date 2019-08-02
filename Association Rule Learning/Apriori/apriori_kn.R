# Apriori

# Data Preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

#create a sparse matrix
#install.packages("arules")
library(arules)

dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#training apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2)) #support = 3products*7days a week/7500 transactions per week

#visualizing the results
inspect(sort(rules, by ='lift')[1:10])