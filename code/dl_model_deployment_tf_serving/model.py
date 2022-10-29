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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

model=Sequential([
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)


_loss, _accuracy= model.evaluate(X_test,y_test)

print(f'Loss & Accuracy:{_loss}, {_accuracy}')

print('*'*10, 'Prediction on new data', '*'*10)

prediction = model.predict(sc.transform(np.array([[40,20000]])))[:, 1]


print(f'for age 40 and salary 40000, prediction is {prediction} ')


# Storing the model 
model.save('../../data/saved_model/dl_model/1')