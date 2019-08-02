#Eclat

#create a sparse matrix
#install.packages("arules")
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2)) #support = 3products*7days a week/7500 transactions per week

#visualizing the results
inspect(sort(rules, by ='support')[1:10])