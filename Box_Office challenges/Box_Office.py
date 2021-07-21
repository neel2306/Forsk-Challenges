import pandas as pd
from sklearn.linear_model import LinearRegression

#Getting the required data.
df = pd.read_csv("Box_Office.csv")

feature = df.iloc[:, 0:1].values
bahubali_label = df.iloc[:, 1:2].values
dangal_label = df.iloc[:, 2:3].values

#Model for bahubali.
regressor1 = LinearRegression()
regressor1.fit(feature, bahubali_label)

#Model for dangal.
regressor2 = LinearRegression()
regressor2.fit(feature, dangal_label)

#Predicting the collection for bahubali2 on the 10th day.
b2_collection = regressor1.predict([[10]])

#Predicting the collection for dangal on the 10th day.
d_collection = regressor2.predict([[10]])

if b2_collection > d_collection:
    print("Bahubali 2 will have more earnings than Dangal on the 10th day.")
else:
    print("Dangal will have more earnings than Bahubali 2 on the 10th day.")