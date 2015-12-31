# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:01:45 2015

@author: Ronan
"""

import timeit, random, sys

import numpy as np

from RBM import RBM

from sklearn.grid_search import ParameterGrid

#data_name='train_data_reduced.npy'
data_name='train_data.npy'
report_name='report'
scoring='accuracy'
do_report = True

# number of epochs allowed without increasing of accuracy
increasing_constraint = 10

params = {'learning_rate':[0.01],
          'training_epochs':[150],
          'batch_size':[20],
          'n_chains':[20],
          'n_hidden':[20000],
          'dropout_rate':[0.5],
          'k':[20]}

param_grid = list(ParameterGrid(params))

hyper_scores = np.zeros(len(param_grid))

for i, current_params in enumerate(param_grid):

    learning_rate = current_params['learning_rate'] 
    training_epochs = current_params['training_epochs']      
    batch_size = current_params['batch_size']
    n_chains = current_params['n_chains']
    n_hidden = current_params['n_hidden']
    dropout_rate = current_params['dropout_rate']
    k = current_params['k']
    
    # Create a report to be saved at the end of execution (when running on the 
    # remote server)
    if do_report:
        report = {"learning_rate":learning_rate,
                  "training_epochs":training_epochs,
                  "batch_size":batch_size,
                  "n_chains":n_chains,
                  "n_hidden":n_hidden,
                  "k":k,
                  "costs":np.zeros(training_epochs),
                  "accuracy":np.zeros(training_epochs),
                  "training_time":0}
    
    data = np.load(data_name)
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
              dropout_rate=dropout_rate,
              batch_size=batch_size,
              np_rng=rng)
    
    #%%========================================================================
    # Training the RBM
    #==========================================================================    
    max_score = -np.inf
    argmax_score = RBM(n_visible=n_visible,
                       n_labels=n_labels,
                       n_hidden=n_hidden, 
                       dropout_rate=dropout_rate,
                       batch_size=batch_size,
                       np_rng=rng)

    start_time = timeit.default_timer()
    for epoch in xrange(training_epochs):
        epoch_time = timeit.default_timer()
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            rbm.update(np_train_set[batch_index*batch_size:(batch_index+1)*batch_size,:], persistent=True, k=k)
            sys.stdout.write("\rEpoch advancement: %d%%" % (100*float(batch_index)/n_train_batches))
            sys.stdout.flush()
        sys.stdout.write("\rEvaluating accuracy...")
        sys.stdout.flush()
        cv_time = timeit.default_timer()
        acc = rbm.cv_accuracy(np_test_set)
        sys.stdout.write('''\rEpoch %i took %f minutes, 
                         accuracy (computed in %f minutes) is %f.\n'''
            % (epoch, 
               ((cv_time-epoch_time) / 60.), 
               ((timeit.default_timer()-cv_time) / 60.),
               acc))
            
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
        if count>=increasing_constraint:
            break
        if do_report:
            report["costs"][epoch] = np.mean(mean_cost)
            report["accuracy"][epoch] = acc
            
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    report['training_time'] = training_time
    print ('Training took %f minutes' % (training_time / 60.))
    if do_report:
        np.save('reports/'+report_name+'_'+str(i), report) 
    
    hyper_scores[i] = max_score       

np.save('reports/hyper_scores', hyper_scores)          
best_params = param_grid[np.argmax(hyper_scores)]
np.save('reports/best_params', best_params)


