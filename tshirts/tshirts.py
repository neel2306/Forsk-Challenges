# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:57:27 2021

@author: Neelabh
"""

#Importing libraries.
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

#Getting the data.
data = pd.read_csv('tshirts.csv')

#Additional information.
'''
shape        -> (85, 3)
columns      -> ['name', 'height (inches)', 'weight (pounds)']
missing data -> name               False
                height (inches)    False
                weight (pounds)    False
                dtype: bool
sample data ->       name       height (inches)  weight (pounds)
                80    Cesar     70.0              155
                81   Ushnik     69.0              140
                82   Hassan     70.0              145
                83   Milson     72.0              166
                84  Shitong     69.0              150
'''

#Declaring features.
features = data.iloc[:,[1,2]].values

#Scaling the data.
scaler = StandardScaler()
features = scaler.fit_transform(features)

#Getting optimum cluster number.
wcss = []
for cluster in range(1,11):
    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

#Elbow method.
plt.plot(range(1,11), wcss)
plt.xlabel('Cluster Number')
plt.ylabel('KMeans value')
plt.title('Elbow Method')
plt.show()

'''
Elbow at 3 => n_clusters = 3
'''

#Clustering the data.
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
pred = kmeans.fit_predict(features)

#Plotting the data.
plt.scatter(features[pred == 0, 0], features[pred == 0, 1], s = 50, c = 'cyan', label = 'Medium')
plt.scatter(features[pred == 1, 0], features[pred == 1, 1], s = 50, c = 'magenta', label = 'Large')
plt.scatter(features[pred == 2, 0], features[pred == 2, 1], s = 50, c = 'black', label = 'Small')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'lightgreen', label = 'Centroids')
plt.title('Clusters of T-Shirts')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()











