import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Getting the data and additional information.
data = pd.read_csv('bluegills.csv')
data.shape
data.columns.tolist()
data.isnull().any(axis = 0)
data.head()

#Declaring our features and labels.
features = data.iloc[:, 0:1].values
labels = data.iloc[:, 1:2].values

#1
#Creating an object for Linear Regression.
regressor = LinearRegression()

regressor.fit(features, labels)

#Plotting.
features_sorted = np.arange(min(features), max(features), 0.1)
features_sorted = features_sorted.reshape(len(features_sorted), 1)
plt.scatter(features, labels)
plt.plot(features_sorted, regressor.predict(features_sorted), color = 'green')
plt.xlabel('Age')
plt.ylabel('Length in mm')
plt.title('Bluegills : Linear Regression Model')
plt.show()

#Polynomial Regression.

#Declaring the object and tranforming features.
regressor_higher = PolynomialFeatures(degree = 2)
features_higher = regressor_higher.fit_transform(features)

#Declaring Linear Regression object again and fitting the features.
regressor_higher_degree = LinearRegression()
regressor_higher_degree.fit(features_higher, labels)


#Plotting.
features_sorted = np.arange(min(features), max(features), 0.1)
features_sorted = features_sorted.reshape(len(features_sorted), 1)
plt.scatter(features, labels, color = 'green')
plt.plot(features_sorted, regressor_higher_degree.predict(regressor_higher.fit_transform(features_sorted)), color = 'magenta')
plt.xlabel('Age')
plt.ylabel('Length in mm')
plt.title('Bluegills : Polynomial Regression Model')
plt.show()

#2

#Predicting with Linear Regression.
print ("Predicting result with Linear Regression :"+str(regressor.predict([[5]])))

#Predicting with Polynomial Regression.
print ("Predicting result with Polynomial Regression :"+str(regressor_higher_degree.predict(regressor_higher.fit_transform([[5]]))))

'''
Predicting result with Linear Regression :[[174.21045499]]
Predicting result with Polynomial Regression :[[165.90231606]]
'''
