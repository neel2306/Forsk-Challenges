import pandas as pd
from sklearn.model_selection import train_test_split

#Getting the required data.
df = pd.read_csv('cars.csv')

#Splitting the data into train module and test module.
df_train, df_test = train_test_split(df, train_size = 0.5, random_state = 0) 

#Writing th train data and test data into separate csv files.
df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')