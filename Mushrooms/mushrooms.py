import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Getting the required data and additional data.
data = pd.read_csv('mushrooms.csv')
data.shape
data.isnull().any(axis = 0)
data.columns.tolist()
data.head()

#Declaring our features and labels.
#Features are odor, population and habitat.
features = data.iloc[:, [5, -1, -2]].values
#Labels is class.
labels = data.iloc[:, 0:1].values

#Transforming our labels.
l_transformer = LabelEncoder()
labels = l_transformer.fit_transform(labels)

#Transforming our features.
cTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,2])], remainder = 'passthrough')
features = cTransformer.fit_transform(features).toarray()

#Splitting train and test data.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size = 0.2, random_state = 0)

#Using KNN classifier.
classifier = KNeighborsClassifier(n_neighbors = 8, p = 1)
classifier.fit(features_train, labels_train)

#Getting the prediction.
prediction = classifier.predict(features_test)

#Comparing data.
compare_data = confusion_matrix(labels_test, prediction)

model_score = round(classifier.score(features_test,labels_test), 3) * 100 
#Result.
print("Accuracy of the model -> " + str(model_score) +'%')

'''
Accuracy of the model -> 99.5%
'''
