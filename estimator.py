# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:26:21 2015

@author: Ronan
"""
import timeit

import numpy as np

import theano
import theano.tensor as T

from RBM import RBM
from load_data import shared_dataset

from theano.tensor.shared_randomstreams import RandomStreams

import os

class Estimator:
                  
    def __init__(self,
                 n_labels = 20,
                 learning_rate=0.01,
                 training_epochs=10,
                 batch_size=20,
                 n_chains=20,
                 n_samples=10,
                 n_hidden=2,
                 k=15,
                 do_report = True,
                 report_folder='reports',
                 report_name='report',
                 scoring='accuracy'):
                     
        self.n_labels = n_labels
        self.learning_rate=learning_rate
        self.training_epochs=training_epochs
        self.batch_size=batch_size
        self.n_chains=n_chains
        self.n_samples=n_samples
        self.n_hidden=n_hidden
        self.k=k
        self.do_report=do_report
        self.report_folder=report_folder
        self.report_name=report_name
        self.scoring=scoring
    
    def get_params():
        pass
    
    def set_params(self,
                   learning_rate=0.01,
                   training_epochs=10,
                   batch_size=20,
                   n_chains=20,
                   n_samples=10,
                   n_hidden=2,
                   k=15):
                       
        self.learning_rate=learning_rate
        self.training_epochs=training_epochs
        self.batch_size=batch_size
        self.n_chains=n_chains
        self.n_samples=n_samples
        self.n_hidden=n_hidden
        self.k=k
    
    def fit(self, X, Y):
        # Create a report to be saved at the end of execution 
        # (when running on the remote server)
        if self.do_report:
            report = {"learning_rate":self.learning_rate,
                      "training_epochs":self.training_epochs,
                      "batch_size":self.batch_size,
                      "n_chains":self.n_chains,
                      "n_samples":self.n_samples,
                      "n_hidden":self.n_hidden,
                      "k":self.k,
                      "costs":np.zeros(self.training_epochs),
#                      "accuracy":np.zeros(self.training_epochs),
                      "pretraining_time":0}
                      
        train_data = np.hstack([Y,X])
        
        n_visible = train_data.shape[1]
        
        # Building of theano format datasets
        train_set = shared_dataset(train_data)
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set.get_value(borrow=True).shape[0] / \
            self.batch_size
        
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data
        
        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(np.zeros((self.batch_size, 
                                                   self.n_hidden),
                                                  dtype=theano.config.floatX),
                                         borrow=True)
        
        # construct the RBM class
        self.rbm = RBM(input=x,
                       n_visible=n_visible,
                       n_labels=self.n_labels,
                       n_hidden=self.n_hidden, 
                       np_rng=rng, 
                       theano_rng=theano_rng)
        
        # get the cost and the gradient corresponding to one step of CD-k
        cost, updates = self.rbm.get_cost_updates(lr=self.learning_rate,
                                                  persistent=persistent_chain, 
                                                  k=self.k)
                                             
#        accuracy = self.rbm.get_cv_error()
        
        #%%====================================================================
        # Training the RBM
        #======================================================================
        
        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set[index * self.batch_size: \
                            (index + 1) * self.batch_size]
            },
            name='train_rbm'
        )
        
        start_time = timeit.default_timer()
    
        max_score = -np.inf
        argmax_score = RBM(input=x,
                           n_visible=n_visible,
                           n_labels=self.n_labels,
                           n_hidden=self.n_hidden, 
                           np_rng=rng, 
                           theano_rng=theano_rng)
#        count = 0
        
        ## go through training epochs
        for epoch in xrange(self.training_epochs):
        
            # go through the training set
            mean_cost = []
            for batch_index in xrange(n_train_batches):
                mean_cost += [train_rbm(batch_index)]
                
            print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)
            
            score = np.mean(mean_cost)

            if score>max_score:
                max_score = score
                argmax_score.clone(self.rbm)
            
#            acc = accuracy.eval()
#            
#            if self.scoring=='cost':
#                score = np.mean(mean_cost)
#            elif self.scoring=='accuracy':
#                score = acc
#            else:
#                raise Warning('''scoring must be cost or accuracy, 
#                              set to accuracy''')
#                score = acc
#                
#            if score>max_score:
#                max_score = score
#                argmax_score.clone(self.rbm)
#                count = 0
#            else:
#                count += 1
#            
#            if count>2:
#                break
                
            if self.do_report:
                report["costs"][epoch] = np.mean(mean_cost)
#                report["accuracy"][epoch] = acc
         
        end_time = timeit.default_timer()
        pretraining_time = (end_time - start_time)
        report['pretraining_time'] = pretraining_time   
        
        self.rbm = argmax_score        
        
        if self.do_report:
            try:
                np.save(self.report_folder+'/'+self.report_name, report)
            except OSError:
                os.mkdir(self.report_folder)
                np.save(self.report_folder+'/'+self.report_name, report)
    
    def predict(self, X):
        # make a prediction for an unlablled sample.
        t_unlabelled = T.tensor3("unlabelled")
        # This is not needed only if we want to make predictions from numpy arrays.
        predict = theano.function(
            [t_unlabelled],
            self.rbm.predict(t_unlabelled),
            name='predict'    
        )
        pred,conf = predict([X])
        return pred
                      
    
#%%============================================================================
# Training the RBM
#==============================================================================
data = np.load('train_data.npy')

X = data[:,20:]
Y = data[:,:20]
est = Estimator(training_epochs = 2) 
est.fit(X, Y)
pred = est.predict(X)
