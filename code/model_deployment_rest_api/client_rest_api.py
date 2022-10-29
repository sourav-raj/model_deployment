# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
Create REST API server which will preprocess the data & 
predict based on the trained model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

import json
import requests

url='http://localhost:8000/model'

request_data=json.dumps({'age':40, 'salary':20000})
response=requests.post(url, request_data)
print(response.text)

request_data=json.dumps({'age':[40,20], 'salary':[20000, 10000]})
response=requests.post(url, request_data)
print(response.text)