# -*- coding: utf-8 -*-
# Indentation: Visual Studio Code

'''
predict using the same param of model
 
'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "souravraj.iitbbs@gmail.com"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--distance",help ="distance to city center")
args = parser.parse_args()

class HousePricePredictor:
    
    def __init__(self):
        print("Prediction started")
        self.coeff=72635.89282856
        self.intercept_=610710.0319872361
        self.distance_to_city_center = float(args.distance)
    
    def predict_price(self):
        price = self.intercept_ - self.coeff * self.distance_to_city_center
        print("House price is >>> "+str(price))
        with open("model_output.txt", "w") as f:
            f.write(str(price))
        
if __name__ == "__main__":
    model_instance = HousePricePredictor()
    model_instance.predict_price()