# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:39:38 2015

@author: navrug
"""


import numpy as np

from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

learning_rate=0.01
training_epochs=10
batch_size=20
n_hidden=200

###############################################################################
# Setting up

data = np.load('train_data_reduced.npy')

n_labels = 20

n_visible = data.shape[1]

# Split of train_data for cross-validation
n_fold = 3
test_n = int(data.shape[0]/n_fold)

permutation = np.random.permutation(data.shape[0])

test_idx = permutation[:test_n]
np_test_set = data[test_idx,:]

train_idx = permutation[test_n:]
np_train_set = data[train_idx,:]
del data


# Load Data

X_train, Y_train = np_train_set[:,n_labels:], np.argmax(np_train_set[:,:n_labels], axis=1)
X_test, Y_test = np_test_set[:,n_labels:], np.argmax(np_test_set[:,:n_labels], axis=1)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = learning_rate
rbm.n_iter = training_epochs
rbm.batch_size = batch_size
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = n_hidden
logistic.C = 1000.0

# Training RBM-Logistic Pipeline
classifier.fit(np_train_set[:,n_labels:], Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

logistic_classifier.score(X_test, Y_test)
classifier.score(X_test, Y_test)
