# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:51:03 2021

@author: Neelabh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('crime_data.csv')

features = dataset.iloc[:, [1,2,4]].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

features = sc.fit_transform(features)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

features = pca.fit_transform(features)


# How much is the loss and how much we are able to retain the information         
explained_variance = pca.explained_variance_ratio_
print(explained_variance)  #[0.7861934  0.15268378]
# first paramater (PC1) is holding 78% of the 3D data
# second parameter (PC2) is holding 15% of the 3D data




#Elbow mathod
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)  # we have not used the fit_predict
    #print("Cluster " + str(i) +  "  =  " + str(kmeans.inertia_))
    wcss.append(kmeans.inertia_)     # ( calculates wcss for a cluster )
    
print(wcss)

#Now plot it        
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#Using Kmeans Clustering

from sklearn.cluster import KMeans
# Since we have seen the visual, we have told the algo to make 4 cluster
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 

print(pred_cluster)

#new add column
dataset["pred_cluster"] = pred_cluster
