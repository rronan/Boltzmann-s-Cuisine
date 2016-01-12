# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:56:53 2016

@author: Ronan
"""

from sklearn.linear_model import LogisticRegression as lr
import numpy as np
from preprocessing import create_submission
from RBM import RBM
from preprocessing import read_data
from sklearn.preprocessing import LabelEncoder

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
load_saved = True

train_data = np.load('train_data.npy')

if load_saved:
    report = np.load("report.npy").item()    
    rbm = RBM(len(train_data), report["n_hidden"], report["batch_size"])
    rbm.W = report["W"]
    rbm.hbias = report["hbias"]
    rbm.vbias = report["vbias"]

Y = np.argmax(train_data[:,:20], axis=1)
train_data = train_data[:,20:]
X = sigmoid(np.dot(train_data, rbm.W) + rbm.hbias)
#X = train_data


classifier = lr(0.01, solver = 'lbfgs', multi_class='multinomial')
classifier.fit(X, Y)

test_data = np.load('test_data.npy')
test_X = sigmoid(np.dot(test_data, rbm.W) + rbm.hbias)
#test_X = test_data

pred = classifier.predict(test_X)
train_ids, train_cuisines, train_ingredients = read_data('train.json')
test_ids, test_cuisines, test_ingredients = read_data('test.json')
del train_ids, train_ingredients, test_cuisines, test_ingredients
le = LabelEncoder()
le.fit(train_cuisines)
pred = le.inverse_transform(pred)
create_submission(test_ids, pred)






