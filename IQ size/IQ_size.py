import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Getting the data and additional information.
data = pd.read_csv('IQ_Size.csv')
data.shape
data.columns.tolist()
data.head()

#Declaring features and labels.
features = data.iloc[:, 1:].values
labels = data.iloc[:, 0].values

#Splitting data into test and train data sets.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size = 0.7, random_state = 0)

#Creating an pbject for linear regression.
regressor = LinearRegression()

#Finding the best fit line.
regressor.fit(features_train, labels_train)

#1
'''
To find the IQ of an individual having brain size of 90, height of 70 inches, and weight 150 pounds'''

pred = regressor.predict([[90, 70, 150]])

print("The IQ of a person having brain size of 90, height of 70 inches, and weight 150 pounds is found to be", int(pred))

'''
Is found to be 106.4381707 which is rounded off to 106
'''

#2 
'''
To predict if a person's brain size and body size (Height and weight) predictive of his or her intelligence
'''
#Using backward elimination method.

#Adding a constant to our features.
features = sm.add_constant(features)
features_ols = features[:, [0,1,2,3]]

#Finding pvalues.
regressor_ols = sm.OLS(labels, features_ols).fit()
print(regressor_ols.summary())

#Removing x3 (weight) column as it's p value is > 5%.
features_ols = features[:, [0,1,2]]
regressor_ols = sm.OLS(labels, features_ols).fit()
print(regressor_ols.summary())

#Removing constant column.
features_ols = features[:, [1,2]]
regressor_ols = sm.OLS(labels, features_ols).fit()
print(regressor_ols.summary())

#Removing x2 (height) column as it's p value is > 5%.
features_ols = features[:, [1]]
regressor_ols = sm.OLS(labels, features_ols).fit()
print(regressor_ols.summary())

print("\nBrain Size is the only factor that helps calculate intelligence of a person")