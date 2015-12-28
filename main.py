# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 16:03:09 2015

@author: Gurvan
"""

import timeit

import numpy as np

from RBM import RBM

import random

np.random.seed(seed=0)
 
 
learning_rate=0.01
training_epochs=10
batch_size=20
n_chains=20
output_folder='rbm_plots'
n_hidden=200
k=5
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
              "k":k,
              "costs":np.zeros(training_epochs),
              "accuracy":np.zeros(training_epochs),
              "pretraining_time":0}

data = np.load('train_data_reduced.npy')

n_labels = 20

n_visible = data.shape[1]

# Split of train_data for cross-validation
n_fold = 3
test_n = int(data.shape[0]/n_fold)

random.seed(0)
permutation = np.random.permutation(data.shape[0])

test_idx = permutation[:test_n]
np_test_set = data[test_idx,:]

train_idx = permutation[test_n:]
np_train_set = data[train_idx,:]
del data

# compute number of minibatches for training, validation and testing
n_train_batches = len(np_train_set) / batch_size


rng = np.random.RandomState(123)


# construct the RBM class
rbm = RBM(n_visible=n_visible,
          n_labels=n_labels,
          n_hidden=n_hidden, 
          batch_size=batch_size,
          np_rng=rng)
          
#%%============================================================================
# Training the RBM
#==============================================================================


plotting_time = 0.
start_time = timeit.default_timer()

## go through training epochs
for epoch in xrange(training_epochs):
    epoch_time = timeit.default_timer()
    # go through the training set
    mean_cost = []
    for batch_index in xrange(n_train_batches):
        rbm.update(np_train_set[batch_index*batch_size:(batch_index+1)*batch_size,:], persistent=True, k=10)
        if np.sum(np.isnan(rbm.W)) > 0:
            print "stop"
    print ('Epoch took %f minutes' % ((timeit.default_timer()-epoch_time) / 60.))
    #print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)
    acc = rbm.cv_accuracy(np_test_set)
    print 'Training epoch', epoch, 'accuracy is', acc
    if do_report:
        report["costs"][epoch] = np.mean(mean_cost)
        report["accuracy"][epoch] = acc
        
 
end_time = timeit.default_timer()

pretraining_time = (end_time - start_time)

print ('Training took %f minutes' % (pretraining_time / 60.))

#%%============================================================================
# Classifying with the RBM
#==============================================================================



if do_report:
    np.save('report', report)
    
#%%============================================================================
# Sampling from the RBM
#==============================================================================


