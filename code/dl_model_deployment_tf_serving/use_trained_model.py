# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
predict new data using the same trained model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

# loading the stored model
from tensorflow.keras.models import load_model
import pickle

trained_model = load_model('../../data/saved_model/dl_model/1')
sc=pickle.load(open('../../data/saved_model/sc.pickle', 'rb'))

prediction = trained_model.predict(sc.transform(np.array([[40,20000]])))[:, 1]


print(f'for age 40 and salary 40000, prediction is {prediction} ')