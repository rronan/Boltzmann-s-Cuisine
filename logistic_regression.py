# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:56:53 2016

@author: Ronan
"""

from sklearn.linear_model import LogisticRegression as lr
import numpy as np
from preprocessing import create_submission

def sigmoid(x):
    return 1/(1+np.exp(-x))
    

# This script needs two object in the environment: rbm, le

data = np.load('train_data_reduced.npy')
Y = np.argmax(data[:,:20], axis=1)
raw_X = data[:,20:]
X = sigmoid(np.dot(raw_X, rbm.W) + rbm.hbias)

classifier = lr(0.1, solver = 'lbfgs', multi_class='multinomial')
classifier.fit(X, Y)

raw_Xtest = np.load('test_data_reduced.npy')
Xtest = sigmoid(np.dot(raw_Xtest, rbm.W) + rbm.hbias)

pred = classifier.predict(Xtest)
test_ids = np.load('test_id.npy')
guess = le.inverse_transform(pred)
final = create_submission(test_ids, guess)






