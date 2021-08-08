import pandas as pd
dataset = pd.read_csv('addhealth.csv')

#Additional info.
dataset.shape
dataset.columns.tolist()
dataset.isnull().any(axis = 0)
dataset.sample(10)

#filling nan values.
#Using mode as the given data is nominal in nature.
for data in dataset:
    dataset[data] = dataset[data].fillna(dataset[data].mode()[0])

#1
#Using logistic regression to find if the adolescent smokes based on the given criteria.

features = dataset[['BIO_SEX','age','WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN',
           'ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1','ESTEEM1']].values
labels = dataset['TREG1'].values

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 0)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(features_train, labels_train)

prediction = regressor.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, prediction)

regressor_score = regressor.score(features_test, labels_test)

print("Model accuracy shown by confusion matrix is -> \n" + str(cm))
print("\nModel accuracy using logistic regression -> " + str(round(regressor_score*100,2))+"\n")
'''
We have got the model score as 87 % (approx) which is a good score on the test data
'''
#2
#Finding if the adolescent is expelled from school on the given criteria.

features2 = dataset[['BIO_SEX', 'VIOL1']].values
labels2 = dataset['EXPEL1'].values

from sklearn.model_selection import train_test_split
f2_train, f2_test, l2_train, l2_test = train_test_split(features2, labels2, random_state = 0)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(f2_train, l2_train)

pred = reg.predict(f2_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(l2_test, pred)

reg_score = reg.score(f2_test, l2_test)

print("Model accuracy shown by confusion matrix is -> \n" + str(cm2))
print("\nModel accuracy using logistic regression -> " + str(round(reg_score*100,2))+"\n")
'''
We have got the model score as 91.3% which is a good score on the test data.
'''