# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:42:41 2019

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

#create the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward")) #ward minimizes the within cluster variance
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

# cluster the data using HC
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)


#Visualizing the clusters - only for 2D
# first we take the value in index 0 which is the X coordinate, 
# and then the value in index 1 which is the y coordinate
# and for data in the first cluster (index 0) we specify y_kmeans == 0
plt.scatter(X[y_hc == 0, 0], X[y_hc ==0, 1], s =100, color ='red', label='Careful') #Careful
plt.scatter(X[y_hc == 1, 0], X[y_hc ==1, 1], s =100, color ='blue', label='Standard') #Standard
plt.scatter(X[y_hc == 2, 0], X[y_hc ==2, 1], s =100, color ='green', label='Target') #Target
plt.scatter(X[y_hc == 3, 0], X[y_hc ==3, 1], s =100, color ='cyan', label='Careless') #Careless
plt.scatter(X[y_hc == 4, 0], X[y_hc ==4, 1], s =100, color ='magenta', label='Sensible') #Sensible

plt.title("Clusters of Clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()