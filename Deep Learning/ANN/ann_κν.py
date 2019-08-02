# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 00:56:49 2019

@author: Infinity
"""

#Installing keras
#conda install -c conda-forge keras in the anaconda prompt

#Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#create dummy variables for categorical with 3
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#remove 1 dummy variable not to fall in the dummy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
#VERY IMPORTANT - WE NEED TO APPLY IT TO EASE CALCULATIONS
#AND WE DO NOT WANT TO HAVE ONE VARIABLE COVERING THE OTHERS
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Lets make the ANN
#Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN with Stochastic Gradient Decent
classifier = Sequential()

#Adding the input layer and the first hidden layer
#Randomly initialize the weights to small number close to 0 (but not 0)
#Input the first observation of your dataset in the input layer, each feature in one input node
#Forward Propagation (higher the value of the activation function for the neuron, more impact the neuron has on the network)
#based on experiments, best function is the rectifier function
#All four are done by the dense function
#for the first hidden layer we need the input dim compulsory = number of independent variables
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim= 11)) #usually we take the mean of the number of nodes in the input layer and the number on the output layer, 11+1/2 = 6 in our case
#add a second hidden layer in the network -imput dim not needed, since it knows to expect input from previous layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
#add the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
##if you are dealing with a dependent variable that has more than two categories like say for
##example three categories then you will need to change two things here.
##First is the output dim parameter that should be set as the number of classes because it will be
##based on the one vs all method while the dependent variable is one hot encoded.
##So here you would need to input three if you have three categories for the independent variable.
##And the second thing that you would need to change is the activation function that in this situation
##would be softmax and softmax is actually the Sigmoid function but applied to a dependent variable
##that has more than two categories. 

#Compiling the ANN 
#adam is the algorithm for the stochastic gradient decent
#since we have a sigmoid function --> logistic regression --. loss function on the logarithmic class
#metrics = criterion used to evaluate the model - typically acc #metrics expects list
classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics=["accuracy"])
##And if your dependent variable has more than two outcomes like three categories then this logarithmic loss
##function is called categorical_crossentropy.
#compare the predicted result to the actual result - measure generated error
#back-propagation update the weight according to how much they are responsible for the error. The learning rate decides by how much we update the weights

#repeat previous steps, either after each observation (reinforcement learning) or after a batch of observations (batch learning)

#when the whole training set passed through the ANN, that makes an epoch, redo more epochs
#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100) #batch size + number of epochs

#Part 3 making the predictions and evaluating the results
#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >0.5) #if y_pred >0 retutn True else return False

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc= (1557+123)/2000