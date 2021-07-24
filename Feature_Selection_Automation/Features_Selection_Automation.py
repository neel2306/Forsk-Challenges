import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

 #Getting the required data.
data = pd.read_csv('Salary_Classification.csv')

#Additional data from the given csv file.
data.shape
data.columns.tolist()
data.head()

#Determining the features and labels.
features = data.iloc[:, 0:4]
labels = data.iloc[:, -1]

#Preparing for encoding the categorical data.
#Creating an object.
ctransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder = 'passthrough') #Using [0] as our categorical data is at the 0th index.

#Converting our features into np.array and tranforming it.
features = np.array(ctransformer.fit_transform(features), dtype = np.float32)

#Dropping redundant data fromm the dummy variables.
features = features[:, 1:]

#Adding constant to our features.
features = sm.add_constant(features)
features_ols = features[:,[0,1,2,3,4,5]]

#automating to remove columns having p value > 5%.
while True:
    regressor_OLS = sm.OLS(endog = labels, exog = features_ols).fit()
    pvalue = regressor_OLS.pvalues
    if pvalue.max() > 0.05:
        features_ols = np.delete(features_ols, pvalue.argmax(), 1)
    else:
        break

print(features_ols.shape)
'''
Shows that only x1 column, which represents the experience column is important in determining the salary
'''








