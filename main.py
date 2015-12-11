# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 16:03:09 2015

@author: Gurvan
"""

import timeit

import numpy as np

import theano
import theano.tensor as T
import os

from RBM import RBM
from load_data import *

import random

from theano.tensor.shared_randomstreams import RandomStreams

learning_rate=0.01
training_epochs=1
batch_size=20
n_chains=20
n_samples=10
output_folder='rbm_plots',
n_hidden=2
k=15
do_report = True

# Create a report to be saved at the end of execution (when running on the 
# remote server)
if do_report:
    report = {"learning_rate":0.01,
              "training_epochs":15,
              "batch_size":20,
              "n_chains":20,
              "n_samples":10,
              "output_folder":'rbm_plots',
              "n_hidden":20,
              "k":15,
              "costs":np.zeros(training_epochs),
              "pretraining_time":0}

data = np.load('train_data.npy')
n_labels = 20

n_visible = data.shape[1]

# Split of train_data for cross-validation
n_fold = 3
test_n = int(data.shape[0]/n_fold)

random.seed(0)
permutation = np.random.permutation(data.shape[0])

test_idx = permutation[range(test_n)]
test_set = data[test_idx,:]

train_idx = permutation[-np.array(range(test_n))]
train_set = data[train_idx,:]

# Building of theano format datasets
train_set, test_set = load_data(train_set, test_set)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# initialize storage for the persistent chain (state = hidden
# layer of chain)
persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                           dtype=theano.config.floatX),
                                 borrow=True)

# construct the RBM class
rbm = RBM(input=x, 
          n_visible=n_visible,
          n_labels=n_labels,
          n_hidden=n_hidden, 
          np_rng=rng, 
          theano_rng=theano_rng)

# get the cost and the gradient corresponding to one step of CD-15
cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                     persistent=persistent_chain, 
                                     k=k)
                                     
# make a prediction for an unlablled sample.
t_unlabelled = T.tensor3("unlabelled")
label, confidence = rbm.predict(t_unlabelled)

#%%============================================================================
# Training the RBM
#==============================================================================


# it is ok for a theano function to have no output
# the purpose of train_rbm is solely to update the RBM parameters
train_rbm = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set[index * batch_size: (index + 1) * batch_size]
    },
    name='train_rbm'
)

plotting_time = 0.
start_time = timeit.default_timer()

## go through training epochs
for epoch in xrange(training_epochs):

    # go through the training set
    mean_cost = []
    for batch_index in xrange(n_train_batches):
        mean_cost += [train_rbm(batch_index)]
    print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)
    if do_report:
        report["costs"][epoch] = np.mean(mean_cost)
        
 
end_time = timeit.default_timer()

pretraining_time = (end_time - start_time)

print ('Training took %f minutes' % (pretraining_time / 60.))

#%%============================================================================
# Classifying with the RBM
#==============================================================================

# predict is used to label test samples.
predict = theano.function(
    [t_unlabelled],
    rbm.predict(t_unlabelled),
    name='predict'    
)

test = data[:,20:]
pred = predict(numpy.array([test]))

if do_report:
    np.save('report.csv', report)
    
#%%============================================================================
# Sampling from the RBM
#==============================================================================

# find out the number of test samples
number_of_test_samples = test_set.get_value(borrow=True).shape[0]

# pick random test examples, with which to initialize the persistent chain
test_idx = rng.randint(number_of_test_samples - n_chains)
persistent_vis_chain = theano.shared(
    np.asarray(
        test_set.get_value(borrow=True)[test_idx:test_idx + n_chains],
        dtype=theano.config.floatX
    )
)


plot_every = 1000
# define one step of Gibbs sampling (mf = mean-field) define a
# function that does `plot_every` steps before returning the
# sample for plotting
(
    [
        presig_hids,
        hid_mfs,
        hid_samples,
        presig_vis,
        vis_mfs,
        vis_samples
    ],
    updates
) = theano.scan(
    rbm.gibbs_vhv,
    outputs_info=[None, None, None, None, None, persistent_vis_chain],
    n_steps=plot_every
)

# add to updates the shared variable that takes care of our persistent
# chain :.
updates.update({persistent_vis_chain: vis_samples[-1]})
# construct the function that implements our persistent chain.
# we generate the "mean field" activations for plotting and the actual
# samples for reinitializing the state of our persistent chain
sample_fn = theano.function(
    [],
    [
        vis_mfs[-1],
        vis_samples[-1]
    ],
    updates=updates,
    name='sample_fn'
)

# TODO: GENERATE NICE RECIPES (e.g. mash + nuts + garlic)
