# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
predict new data using the same trained model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

import numpy as np
import pickle

log_reg=pickle.load(open('../../data/saved_model/log_reg.pickle', 'rb'))
sc=pickle.load(open('../../data/saved_model/sc.pickle', 'rb'))

print("=="*30)
print('*'*10, 'Prediction on new data', '*'*10)

prediction = log_reg.predict(sc.transform(np.array([[40,20000]])))

prediction_proba = log_reg.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

print(f'for age 40 and salary 40000, prediction is {prediction} with prob {prediction_proba}')

