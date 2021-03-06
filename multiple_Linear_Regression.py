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

#Fitting Multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressior = LinearRegression()
regressior.fit(x_train,y_train)

#Predicting
y_pred = regressior.predict(x_test)

# Optical Model using Backward Elemenation

import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values =x, axis =1)
x_opt = x[:, [0,1,2,3,4,5]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS  = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()