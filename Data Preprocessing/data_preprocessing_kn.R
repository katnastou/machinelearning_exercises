#Data Preprocessing

dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age), #if statement, if value is missing
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), #if true run function to calcuclate mean from existing values
                     dataset$Age)#else take existing value

dataset$Salary = ifelse(is.na(dataset$Salary), #if statement, if value is missing
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), #if true run function to calcuclate mean from existing values
                     dataset$Salary)#else take existing value

#Encoding Categorical data

dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3)
                         )

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('Yes', 'No'),
                         labels = c(1,0))

#Splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[,2:3] = scale(training_set[,2:3]) # WE CHANGED THE VALUES WITH FORMAT SO NOW IT IS NOT NUMERIC
test_set[,2:3]= scale(test_set[,2:3])
