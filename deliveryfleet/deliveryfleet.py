# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:51:45 2021

@author: Neelabh
"""

#Importing libraries.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

#Getting the data.
data = pd.read_csv('deliveryfleet.csv')

#Getting additional data.
print(data.shape,'\n', data.columns.tolist(),'\n', data.isnull().any(axis = 0), '\n', data.head())

'''
(4000, 3) 
 ['Driver_ID', 'Distance_Feature', 'Speeding_Feature'] 
 Driver_ID           False
Distance_Feature    False
Speeding_Feature    False
dtype: bool 
     Driver_ID  Distance_Feature  Speeding_Feature
0  3423311935             71.24              28.0
1  3423313212             52.53              25.0
2  3423313724             64.54              27.0
3  3423311373             55.69              22.0
4  3423310999             54.58              25.0
'''

#Declaring the features and converting it to ndarray.
features = data.iloc[:,[1,2]].values

#Visualizing the features.
x = features[:,0]
y = features[:,1]
plt.title('Delivery Fleet')
plt.ylabel('Speed Feature')
plt.xlabel('Distance Feature')
plt.scatter(x,y)

#Getting the optimum wcss value.
wcss = []
for cluster in range(1, 11):
    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

#Plotting the wcss.
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

'''
The elbow is formed at 2. So the optimum number of clusters for this dataset will be 2
'''


#Fitting the data.
k_model = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
pred = k_model.fit_predict(features)

#Visualising the clusters
plt.scatter(features[pred == 0, 0], features[pred == 0, 1], s = 50, c = 'black', label = 'Rural Drivers')
plt.scatter(features[pred == 1, 0], features[pred == 1, 1], s = 50, c = 'yellow', label = 'Urban Drivers')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'cyan', label = 'Centroids')
plt.title('Clusters of Drivers')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.legend()
plt.show()

#Clustering data for rash/safe driving.
k_model = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
pred = k_model.fit_predict(features)

#Visualising the data.
plt.scatter(features[pred == 0, 0], features[pred == 0, 1], s = 50, c = 'green', label = 'Rural Safe Drivers')
plt.scatter(features[pred == 1, 0], features[pred == 1, 1], s = 50, c = 'pink', label = 'Urban Safe Drivers')
plt.scatter(features[pred == 2, 0], features[pred == 2, 1], s = 50, c = 'cyan', label = 'Urban Rash Drivers')
plt.scatter(features[pred == 3, 0], features[pred == 3, 1], s = 50, c = 'magenta', label = 'Rural Rash Drivers')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Drivers')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.legend()
plt.show()







