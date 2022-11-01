# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
model creation 
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

house_price=pd.read_csv('../../data/houseprice.csv')
print(house_price.shape)
print(house_price.describe())

X=house_price.iloc[:, :-1].values
y=house_price.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lin_reg=LinearRegression()

lin_reg.fit(X_train, y_train)

rsquared = lin_reg.score(X_test, y_test)
print(f'R2: {rsquared}')

intercept_ = lin_reg.intercept_
coefficient_ = lin_reg.coef_
print(f'Coefficient & intercept of model are: {coefficient_}, {intercept_}')

y_pred = lin_reg.predict(X_test)

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, y_pred, color = 'green')
plt.title('House price analysis')
plt.xlabel('Distance to City center')
plt.ylabel('House price USD')
plt.show()

print("=="*30)
print('*'*10, 'Prediction on new data', '*'*10)

prediction = lin_reg.predict([[2.5]])

print(f'for distance of 2.5, prediction is {prediction}')




















