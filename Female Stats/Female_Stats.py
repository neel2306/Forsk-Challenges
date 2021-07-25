import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('Female_Stats.csv')

#Getting additional information on the dataframe.
data.shape
data.columns.tolist()
data.head()
data.isnull().any(axis = 0)

#Declaring features and labels.
features = np.array(data.iloc[:, 1:])
labels = data.iloc[:,0]

#Splitting data into train set and test set.
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#Creating our linear model.
regressor = LinearRegression()
regressor.fit(features_train,labels_train)

#Adding constant to our feature_train.
features = sm.add_constant(features)

#1
#Finding the p values.
features_ols = features[:,[0,1,2]]
while True:
    regressor_OLS = sm.OLS(endog = labels, exog = features_ols).fit()
    pvalue = regressor_OLS.pvalues
    if pvalue.max() > 0.05:
        features_ols = np.delete(features_ols, pvalue.argmax(), 1)
    else:
        break
    
features_ols.shape
'''
Implies that the student's dads and moms height are an important factor in determining the students height as well.
'''

#2
print("When Father's Height is Held Constant then the average height increase by",regressor_OLS.params[1])

#3
print("When Mother's Height is Held Constant then the average height increase by",regressor_OLS.params[2])

'''
The output gaines for 2 and 3 are
When Father's Height is Held Constant then the average height increase by 0.3035055794072806
When Mother's Height is Held Constant then the average height increase by 0.38785558789574814
respectively.
'''