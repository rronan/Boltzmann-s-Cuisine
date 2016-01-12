# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:25:34 2016

@author: navrug
"""

import timeit

import numpy as np
import sys 
from RBM import RBM
from sklearn.linear_model import LogisticRegression

import random

np.random.seed(seed=0)
 
 
learning_rate=0.01
training_epochs=50
batch_size=20
n_chains=20
output_folder='rbm_plots'
n_hidden=2000
dropout_rate=0.5
k=20
do_report = True

# Create a report to be saved at the end of execution (when running on the 
# remote server)

if do_report:
    report = {"learning_rate":learning_rate,
              "training_epochs":training_epochs,
              "batch_size":batch_size,
              "n_chains":n_chains,
              "output_folder":'rbm_plots',
              "n_hidden":n_hidden,
              "dropout_rate":dropout_rate,
              "k":k,
              "costs":np.zeros(training_epochs),
              "accuracy":np.zeros(training_epochs),
              "pretraining_time":0}

data = np.load('train_data.npy')

n_labels = 20

n_visible = data.shape[1]-n_labels

# Split of train_data for cross-validation
n_fold = 3
test_n = int(data.shape[0]/n_fold)

random.seed(0)
permutation = np.random.permutation(data.shape[0])

test_idx = permutation[:test_n]
test_set = data[test_idx,:]

train_idx = permutation[test_n:]
train_set = data[train_idx,:]
del data

test_labels = np.argmax(test_set[:,:n_labels], axis=1)
train_labels = np.argmax(train_set[:,:n_labels], axis=1)

# compute number of minibatches for training, validation and testing
batches = [train_set[i:i + batch_size,n_labels:] \
    for i in range(0, train_set.shape[0], batch_size)]

rng = np.random.RandomState(123)
# construct the RBM class
rbm = RBM(n_visible=n_visible,
          n_hidden=n_hidden, 
          dropout_rate=dropout_rate,
          batch_size=batch_size,
          np_rng=rng)
          
          
#%%============================================================================
# Training the RBM
#==============================================================================


start_time = timeit.default_timer()
accuracies = []
argmax_acc = 0
for epoch in xrange(training_epochs):
    epoch_time = timeit.default_timer()
    mean_cost = []
    for batch_index, batch in enumerate(batches):
        rbm.update(batch, persistent=True, k=k)
        sys.stdout.write("\rEpoch advancement: %d%%" % (100*float(batch_index)/len(batches)))
        sys.stdout.flush() 
    # Training Logistic regression
    sys.stdout.write("\rTraining softmax...")
    sm_time = timeit.default_timer()
    softmax_classifier = LogisticRegression(penalty='l1', 
                                            C=1.0, 
                                            solver='lbfgs', 
                                            multi_class='multinomial')
    softmax_classifier.fit(rbm.propup(train_set[:,n_labels:], np.ones((len(train_set),n_hidden))),
                           train_labels)
    sys.stdout.write('\rSoftmax trained in %f minutes.\n' % ((timeit.default_timer()-sm_time) / 60.))
    sys.stdout.write("Evaluating accuracy...")
    cv_time = timeit.default_timer()
    acc = softmax_classifier.score(rbm.propup(test_set[:,n_labels:], np.ones((len(test_set),n_hidden))), 
                                   test_labels)
    accuracies.append(acc)
    sys.stdout.write('''\rEpoch %i took %f minutes, 
                     accuracy (computed in %f minutes) is %f.\n'''
        % (epoch, ((cv_time-epoch_time) / 60.), 
           ((timeit.default_timer()-cv_time) / 60.), acc))
    if do_report:
        report["costs"][epoch] = np.mean(mean_cost)
        report["accuracy"][epoch] = acc
        if (acc>argmax_acc):
            report["W"] = rbm.W
            report["hbias"] = rbm.hbias
            report["vbias"] = rbm.vbias
            np.save('report', report)
            sys.stdout.write("Model saved \n")
        
end_time = timeit.default_timer()
pretraining_time = (end_time - start_time)
report["pretraining_time"] = pretraining_time
print ('Training took %f minutes' % (pretraining_time / 60.))

if do_report:
    np.save('report', report)

#%%============================================================================
# Classifying with the RBM
#==============================================================================


    
#%%============================================================================
# Sampling from the RBM
#==============================================================================


