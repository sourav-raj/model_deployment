# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
Create REST API server which will preprocess the data & 
predict based on the trained model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

import numpy as np
from flask import Flask, request
import pickle

app=Flask(__name__)

model_trained=pickle.load(open('../../data/saved_model/log_reg.pickle', 'rb'))
sc=pickle.load(open('../../data/saved_model/sc.pickle', 'rb'))

@app.route('/model', methods=['POST'])
def model():
    request_data=request.get_json(force=True)
    age=request_data['age']
    salary=request_data['salary']
    if type(age)==list:
        age=np.array(age)
        salary=np.array(salary)
        prediction = model_trained.predict(sc.transform(np.stack((age, salary),axis=1)))

        prediction_proba = model_trained.predict_proba(sc.transform(np.stack((age, salary),axis=1)))[:,1]
    else:
        prediction = model_trained.predict(sc.transform(np.array([[age,salary]])))

        prediction_proba = model_trained.predict_proba(sc.transform(np.array([[age,salary]])))[:,1]

    return f'for age {age} and salary {salary}, prediction is {prediction} with prob {prediction_proba}'

if __name__=='__main__':
    app.run(port=8000, debug=True)
