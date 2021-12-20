# -*- coding: utf-8 -*-

#Libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data preprocessing

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

#Encoding Catagorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Avoiding dummy variable trap

x = x[:,1:]

#spliting data in train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 1/3, random_state= 0)

