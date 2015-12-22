# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:07:33 2015

@author: Ronan
"""

import timeit

import numpy as np

import theano
import theano.tensor as T

from RBM import RBM
from load_data import load_data

import random

from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.grid_search import ParameterGrid

import os

report_folder='reports'
report_name='report'
scoring='accuracy'
do_report = True

params = {'learning_rate':[0.01],
          'training_epochs':[10],
          'batch_size':[20],
          'n_chains':[20],
          'n_samples':[10],
          'n_hidden':[2],
          'k':[5, 15]}

param_grid = list(ParameterGrid(params))

hyper_scores = np.zeros(len(param_grid))

i = 0

for current_params in param_grid:

    learning_rate = current_params['learning_rate'] 
    training_epochs = current_params['training_epochs']      
    batch_size = current_params['batch_size']
    n_chains = current_params['n_chains']
    n_samples = current_params['n_samples']
    n_hidden = current_params['n_hidden']
    k = current_params['k']
    
    # Create a report to be saved at the end of execution (when running on the 
    # remote server)
    if do_report:
        report = {"learning_rate":learning_rate,
                  "training_epochs":training_epochs,
                  "batch_size":batch_size,
                  "n_chains":n_chains,
                  "n_samples":n_samples,
                  "n_hidden":n_hidden,
                  "k":k,
                  "costs":np.zeros(training_epochs),
                  "accuracy":np.zeros(training_epochs),
                  "pretraining_time":0}
    
    data = np.load('train_data.npy')
    
    X = data[:,20:]
    Y = data[:,:20]
    
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
    
    # Building of theano format datasets
    train_set, test_set = load_data(np_train_set, np_test_set)
    
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
              validation=test_set,
              n_visible=n_visible,
              n_labels=n_labels,
              n_hidden=n_hidden, 
              np_rng=rng, 
              theano_rng=theano_rng)
    
    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, 
                                         k=k)
    accuracy = rbm.get_cv_error()
                                         
    # make a prediction for an unlablled sample.
    t_unlabelled = T.tensor3("unlabelled")
    label, confidence = rbm.predict(t_unlabelled)
    
    #%%========================================================================
    # Training the RBM
    #==========================================================================
    
    
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
    
    max_score = -np.inf
    argmax_score = RBM(input=x,
                       n_visible=n_visible,
                       n_labels=n_labels,
                       n_hidden=n_hidden, 
                       np_rng=rng, 
                       theano_rng=theano_rng)    
    
    start_time = timeit.default_timer()
    
    ## go through training epochs
    for epoch in xrange(training_epochs):
    
            # go through the training set
            mean_cost = []
            for batch_index in xrange(n_train_batches):
                mean_cost += [train_rbm(batch_index)]
            
            cost = np.mean(mean_cost)
            acc = accuracy.eval()
            
            print 'Training epoch %d, cost is %.3f, accuracy is %.3f'  %(epoch,
                                                                         cost,
                                                                         acc)
            
            if scoring=='cost':
                score = np.mean(mean_cost)
            elif scoring=='accuracy':
                score = acc
            else:
                raise Warning('''scoring must be cost or accuracy, 
                              set to accuracy''')
                score = acc
                
            if score>max_score:
                max_score = score
                argmax_score.clone(rbm)
                count = 0
            else:
                count += 1
            
            if count>2:
                break
                
            if do_report:
                report["costs"][epoch] = np.mean(mean_cost)
                report["accuracy"][epoch] = acc
     
    hyper_scores[i] = max_score
    i +=1
    
    end_time = timeit.default_timer()
    
    pretraining_time = (end_time - start_time)
    
    print ('Training took %f minutes' % (pretraining_time / 60.))
    
    if do_report:
        try:
            np.save(report_folder+'/'+report_name+'_i', report)
        except OSError:
            os.mkdir(report_folder)
            np.save(report_folder+'/'+report_name+'_i', report)
            

np.save(report_folder+'/hyper_scores', hyper_scores)          
            
best_params = param_grid[np.argmax(hyper_scores)]
np.save(report_folder+'/best_params', best_params)
