#Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue") # to create the line we need to put the predictions on the training set on the y axis
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of XP")
plt.ylabel("Salary")
plt.show()

#visualizing the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue") # to create the line we need to put the predictions on the training set on the y axis
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of XP")
plt.ylabel("Salary")
plt.show()