import pandas as pd
import numpy as np
import re 

df = pd.read_csv("thanksgiving.csv", encoding="Windows 1252")
#1
#Fetching column names.
column_data = df.columns.tolist()
no_of_columns = len(column_data)
#So there are 65 columns in the data frame.

#Now to assign a numeric to each column.
column_code = [x for x in range(0,65)]

new_columns = dict(zip(column_code, column_data))

#Altering the column names in the data frame.
df.columns = new_columns 

#2
#Fetching the data of people who celebrate thanksgiving.
people_thanksgiving = df[df[1] == 'Yes']

#3
#Checking and filling for missing values.
df.isnull().any(axis = 0)
df = df.replace(np.nan, 'Missing')

#4
#Finding out required columns.
print(new_columns)
#We have found that region --> 64; income --> 63 and state --> 60.
#Grouping by region.
region_df = df.groupby(64)
print(region_df.size())

#Grouping by income.
income_df = df.groupby(63)
print(income_df.size())

#Grouping by state.
state_df = df.groupby(60)
print(state_df.size())

#To show the relation b/w sauces and incomes.
sauce_df = df.groupby(8)[63].value_counts()
print(sauce_df)

#To filter the gender and assign 1 to female and 0 to male.
def gender_assigner(gender):
    if gender == 'Male':
        gender = 0
    elif gender == 'Female':
        gender = 1
    return gender

df[62] = df[62].apply(gender_assigner)

#Income cleanup using sauces as the base.
df[63] = df[63].replace(['Prefer not to answer', 'Missing'],['0','0'])
reg_ex = re.compile("\d+\W*\d+")

def income_filter(income):
    income = reg_ex.findall(income)
    income = [int(x.replace(",", "")) for x in income]
    return sum(income)/(len(income)+0.1)

df[63] = df[63].apply(income_filter)

income_by_sauce_type = df.groupby(8)[63]
print (income_by_sauce_type.groups)

avg_income_by_sauce_type = income_by_sauce_type.mean()
print (avg_income_by_sauce_type)

#Plotting income vs sauces.
print(avg_income_by_sauce_type.plot.bar())
