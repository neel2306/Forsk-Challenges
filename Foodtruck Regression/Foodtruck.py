import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Getting the required data.
df = pd.read_csv('Foodtruck.csv')

#Creating a model.
features = df.iloc[:,0:1].values
labels = df.iloc[:,1:2].values

#Splitting the data.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 0)

#Creating an object.
regressor = LinearRegression()

#Finding the best fit line.
regressor.fit(features_train, labels_train)

#Reshaping the data into 2-D and predicting the profit.
regressor.predict([[3.073]])

'''
As the profit is predicted is in -ve (-0.38 approx), it is advisable to the restaurant chain to not open a franchise in jaipur city as it will result in a loss.
'''
