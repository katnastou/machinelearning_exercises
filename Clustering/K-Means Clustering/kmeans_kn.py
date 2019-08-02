# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:59:46 2019

@author: Infinity
"""
#%reset -f
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


# Using the elbow method to find the number of clusters
from sklearn.cluster import KMeans
wcss = [] #within clusters sum of squares
for i in range (1,11): #we will do kmeans from 1 to 10 and compute wcss each time
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #inertia_: Sum of squared distances of samples to their closest cluster center.
plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Applying k-means to the mall dataset
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X) #creates an array that assigns each point to a cluster

#Visualizing the clusters
# first we take the value in index 0 which is the X coordinate, 
# and then the value in index 1 which is the y coordinate
# and for data in the first cluster (index 0) we specify y_kmeans == 0
#plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans ==0, 1], s =100, color ='red', label='Cluster 1') 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans ==0, 1], s =100, color ='red', label='Careful') 
#plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans ==1, 1], s =100, color ='blue', label='Cluster 2') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans ==1, 1], s =100, color ='blue', label='Standard') 
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans ==2, 1], s =100, color ='green', label='Cluster 3') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans ==2, 1], s =100, color ='green', label='Target') 
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans ==3, 1], s =100, color ='cyan', label='Cluster 4') 
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans ==3, 1], s =100, color ='cyan', label='Careless') 
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans ==4, 1], s =100, color ='magenta', label='Cluster 5') 
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans ==4, 1], s =100, color ='magenta', label='Sensible') 
#plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, color ='yellow', label='Centroids') 
plt.title("Clusters of Clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
