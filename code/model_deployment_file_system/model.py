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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

product_data=pd.read_csv('../../data/productpurchasedata.csv')
print(product_data.shape)
print(product_data.describe())

X=product_data.iloc[:, :-1].values
y=product_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

log_reg=LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred=log_reg.predict(X_test)
y_prob=log_reg.predict_proba(X_test)[:,1]

_accuracy= accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test, y_pred)
clReport=classification_report(y_test,y_pred)

print(f'Accuracy: {_accuracy}')
print("=="*30)
print("Confusion Matrix")
print(cm)
print("=="*30)
print(clReport)

print("=="*30)
print('*'*10, 'Prediction on new data', '*'*10)

prediction = log_reg.predict(sc.transform(np.array([[40,20000]])))

prediction_proba = log_reg.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

print(f'for age 40 and salary 40000, prediction is {prediction} with prob {prediction_proba}')


# Storing the model and standard scaler
pickle.dump(log_reg, open('../../data/saved_model/log_reg.pickle', 'wb'))
pickle.dump(sc, open('../../data/saved_model/sc.pickle', 'wb'))


















