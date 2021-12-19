# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data preprocessing

data = pd.read_csv('Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

#spliting data in train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 1/3, random_state= 0)

#Fitting data

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(x_train, y_train)

#predicting 
y_pred = regressor.predict(x_test)

#visualizing training set

plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('Salary vs Experence (training set)')
plt.xlabel('Years of Experence')
plt.ylabel('Salary')
plt.show()