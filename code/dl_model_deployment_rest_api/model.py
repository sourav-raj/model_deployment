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

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

review_data=pd.read_csv('../../data/Restaurant_Reviews.tsv', delimiter='\t')
print(review_data.shape)
print(review_data.describe())

ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ',text).lower().split()
    clean_text = [ps.stem(word) for word in text if not word in set([val for val in stopwords.words('english') if val!='not'])]
    clean_text = ' '.join(clean_text)
    return clean_text
review_data['review_clean']=review_data['Review'].apply(preprocess_text)
review_data.head()

tfidf = TfidfVectorizer(max_features = 1500, min_df = 3, max_df = 0.6)

X = tfidf.fit_transform(review_data['review_clean'].values).toarray()
y = review_data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=Sequential([
    Dense(200, activation='relu'),
    Dense(100, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

model.summary()

_loss, _accuracy= model.evaluate(X_test,y_test)

print(f'Loss & Accuracy:{_loss}, {_accuracy}')

print('*'*10, 'Prediction on new text', '*'*10)

text='Good batting by England'

text=tfidf.transform([preprocess_text(text)])

prediction = model.predict(text)[:, 1]


print(f'for given {text}, prediction is {prediction} ')


# Storing the model 
model.save('../../data/saved_model/dl_text_classifier_model/1')
pickle.dump(tfidf, open('../../../data/saved_model/tfidfmodel.pickle','wb'))
pickle.dump(preprocess_text, open('data/saved_model/preprocess_text.pickle','wb'))






