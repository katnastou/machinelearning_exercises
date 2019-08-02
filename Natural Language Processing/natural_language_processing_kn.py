# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:59:47 2019

@author: Katerina
"""
#NLP
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3) #ignore double quotes

#Cleaning the texts
import re
import nltk
#nltk.download('popular')
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', " ", dataset['Review'][i]) #substitute everything but words
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() #take the stem of a word e.g. loved or loving --> love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))] #create a set to get only unique words and the algorithm executes faster
    #make the review a string again
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) #if working with large texts, use max_features to keep only most common words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

'''
#NAIVE BAYES!!!

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
'''
###Decision Tree
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
###RANDOM FOREST
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
acc = (103+67)/200 #NB: 0.73, DT: 0.71, RF: 0.85
#Precision = TP / (TP + FP)
prec = 103/(103+14) #NB: 0.56, DT: 0.76, RF: 0.88
#Recall = TP / (TP + FN)
rec = 103/(103+66) #NB: 0.82, DT: 0.67, RF: 0.61
#F1 score = 2 * Precision * Recall / (Precision + Recall)
F1_NB = (2*0.56*0.82)/(0.56+0.82) #0.66
F1_DT = (2*0.76*0.67)/(0.76+0.67) #0.71
F1_RF = (2*0.88*0.61)/(0.88+0.61) #0.72


#    CART
#    C5.0
#    Maximum Entropy