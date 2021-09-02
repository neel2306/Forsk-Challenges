# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:43:08 2021

@author: Neelabh
"""

import pandas as pd
from apyori import apriori

# Data Preprocessing
# Column names of the first row is missing, header - None
dataset = pd.read_csv('BreadBasket_DMS.csv')


a = dict(dataset["Item"].value_counts())

items = []
values = []

i = 0
for x,y in a.items():
    if i == 15:
        break
    items.append(x)
    values.append(y)
    i += 1

# for pie chart
from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(values, labels = items,autopct='%1.2f%%')
plt.show()


"""

2. Find the associations of items where min support should be 0.0025, 
min_confidence=0.2, min_lift=3.

"""

import pandas as pd
from apyori import apriori

# Data Preprocessing
# Column names of the first row is missing, header - None
dataset = pd.read_csv('BreadBasket_DMS.csv', header = None)

dataset.drop([0],axis=0,inplace=True)

dataset[2] = dataset[2].astype(int)






transaction = []
value = 1
new = []
for j,i in enumerate(dataset[2]):
    if i == value:
        new.append(str(dataset.values[j,3]))
        value = i
    else:
        transaction.append(new)
        new = []
        new.append(str(dataset.values[j,3]))
        value = i
        
    
rules = apriori(transaction, min_support = 0.0025, min_confidence = 0.2, min_lift = 3)


print(type(rules))

# Visualising the results
results = list(rules)
print(len(results))


"""

3. Out of given results sets, show only names of the associated 
item from given result row wise.


"""

for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))












