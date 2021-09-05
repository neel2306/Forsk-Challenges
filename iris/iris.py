# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:54:24 2021

@author: Neelabh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
from sklearn import datasets

iris = datasets.load_iris()

print(iris.DESCR)

#inforation about boston dataset
iris.keys() 
iris.data  # ( features )
iris.data.shape
iris.feature_names
iris.target 

iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df.head()

features = iris_df.iloc[:,0:].values
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

features = pca.fit_transform(features)


# How much is the loss and how much we are able to retain the information
explained_variance = pca.explained_variance_ratio_
print(explained_variance)  #[0.92461872 0.05306648]
# first paramater (PC1) is holding 92% of the 4D data
# second parameter (PC2) is holding 5% of the 4D data




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

from sklearn.cluster import KMeans
# Since we have seen the visual, we have told the algo to make 4 cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 

print(pred_cluster)

iris_df["pred_cluster"] = pred_cluster

iris_df["pred_cluster"].value_counts(dropna=False)