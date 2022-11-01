# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
predict using the same param of model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

# import predictor

# model_object = predictor.HousePricePredictor()

# model_object.predict_price(2)


import os
command = 'python predictor.py --distance 1'
os.system(command)