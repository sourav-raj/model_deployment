# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
Predict new data by the model running on docker using REST API.
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

# loading the stored model
from tensorflow.keras.models import load_model
import pickle
import json
import requests

import numpy as np

url='http://localhost:8501/v1/models/product_purchase_model:model'
sc=pickle.load(open('../../data/saved_model/sc.pickle', 'rb'))
instances=sc.transform(np.array([[40,20000]]))
instances=[[v for v in val] for val in instances]
request_data=json.dumps({"signature_name": "serving_default",
                   "instances":instances})

# prediction = trained_model.predict()[:, 1]
json_response = requests.post(url,request_data)

print(f'for age 40 and salary 40000, prediction is {json_response.text} ')