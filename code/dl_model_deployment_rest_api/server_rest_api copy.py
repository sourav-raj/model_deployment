# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
Create REST API server which will preprocess the text data & 
predict based on the trained model.

'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

import numpy as np
from flask import Flask, request
import pickle
import requests
import tensorflow as tf
from tensorflow.keras.model import load_model

app=Flask(__name__)

model_trained=load_model('../../data/saved_model/dl_text_classifier_model/1')

tfidf=pickle.load(open('../../../data/saved_model/tfidfmodel.pickle','rb'))
preprocess_text=pickle.load(open('data/saved_model/preprocess_text.pickle', 'rb'))

@app.route('/model', methods=['POST'])
def model():
    request_data=request.get_json(force=True)
    text=request_data['sentence']
    
    if type(age)==list:
        text=np.array(text)
        sentences=[[preprocess_text(text)] for s in text]
        text_vector=tfidf.transform(sentences)
        prediction = model.predict(text_vector)[:, 1]

    else:
        text_vector=tfidf.transform([preprocess_text(text)])

        prediction = model.predict(text_vector)[:, 1]
        
    return f'for given text: {text}, prediction is {prediction}'


app.run()
