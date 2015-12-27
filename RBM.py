# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 18:32:44 2015

@author: navrug
"""

"""
Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""

import numpy as np

from scipy.special import expit


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        n_visible,
        n_labels,
        n_hidden,
        batch_size,
        W=None,
        hbias=None,
        vbias=None,
        np_rng=None,
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units
        
        :param n_visible: number of visible units that are labels

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        if np_rng is None:
            # create a number generator
            np_rng = np.random.RandomState(0)

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
        
            # Recurrent error at this line "AttributeError: 'TensorVariable' 
            # object has no attribute 'sqrt'. I try to convert n_visible
            # to int.
        
            # n_visible in a TensorVariable object, see:
            # http://deeplearning.net/software/theano/library/tensor/basic.html
            W = np.asarray(
                    np_rng.uniform(
                        low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                        high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                        size=(n_visible, n_hidden)
                    ),
                    dtype=float
                )
            
        if hbias is None:
            # create shared variable for hidden units bias
            hbias = np.zeros(n_hidden, dtype=float)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = np.zeros(n_visible, dtype=float)
            
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.persistent = None

        
    def clone(self, rbm_object):
        self.n_visible = rbm_object.n_visible
        self.n_labels = rbm_object.n_labels
        self.n_hidden = rbm_object.n_hidden
        self.W = rbm_object.W
        self.hbias = rbm_object.hbias
        self.vbias = rbm_object.vbias
        

    def propup(self, v):
        '''This function propagates the visible units activation upwards to
        the hidden units
        '''
        return expit(np.dot(v, self.W) + self.hbias)


    def sample_h_given_v(self, v):
        ''' This function infers state of hidden units given visible units '''
        h_mean = self.propup(v)
        return np.random.binomial(size=h_mean.shape, n=1, p=h_mean)       


    def propdown(self, h):
        '''This function propagates the hidden units activation downwards to
        the visible units
        '''
        return expit(np.dot(h, self.W.T) + self.vbias)


    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        v_mean = self.propdown(h0_sample)
        return np.random.binomial(size=v_mean.shape, n=1, p=v_mean)


    def gibbs_hvh(self, h):
        v = self.sample_v_given_h(h)
        return self.sample_h_given_v(v)


    def gibbs_vhv(self, v):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        h = self.sample_h_given_v(v)
        return self.sample_v_given_h(h)

    
    def build_prediction_base(self, test_set):
        ''' DEPRECATED: The data we use is too big to allow building this in
            memory. Use cv_accuracy instead.
            This function builds the matrix that is used for efficient
            predicitons. Must be fed with a boolean test set.'''
        self.test_labels = np.argmax(test_set[:, :self.n_labels], axis=1)
        possible_labels = np.repeat(np.eye(self.n_labels, dtype=bool), len(test_set[:,self.n_labels:]), axis=0)
        possible_labels.shape = (self.n_labels,  len(test_set[:,self.n_labels:]), self.n_labels)
        tiled_test_set = np.tile(test_set[:,self.n_labels:],(1,self.n_labels,1))
        tiled_test_set.shape = (self.n_labels,  len(test_set[:,self.n_labels:]), len(test_set[:,self.n_labels:][0]))
        self.prediction_base = np.concatenate((possible_labels, tiled_test_set), axis=2)
        
        
                
    def predict(self):
        ''' DEPRECATED: The data we use is too big to allow building this in
            memory. Use cv_accuracy instead.
        This function makes a prediction for an unlabelled sample,
        this is done by computing Z.P(v_unlablled,label) which is proportional
        to P(label|v_unlabelled).'''
        '''
        P(vlabel|vdata) = P(vlabel, vdata) / P(vdata)
                          = P(vlabel, vdata) / Sum_{vlabel'} P(vlabel', vdata)
                          = Prod_j exp(vbias_j*v_j) * Prod(1+exp(c_i + (Wv)_i))
                            / Sum_{vlabel'} Prod_j exp(vbias_j*v'_j) * Prod(1+exp(c_i + (Wv')_i))
                          = exp(hbias_label) * Prod(1+exp(c_i + (Wv)_i))
                            / Sum_{vlabel'} exp(hbias_label') * Prod(1+exp(c_i + (Wv')_i))
                   prop. to exp(hbias_label) * Prod(1+exp(c_i + (Wv)_i))
        '''
        
        # It may be needed use log of probabilities.
        p =  np.einsum('ij,i->ij',np.prod(1 + np.exp(self.hbias + np.dot(self.prediction_base,self.W)), axis=2),np.exp(self.vbias[:self.n_labels])) 
        labels = np.argmax(p, axis=0)   
        # TODO: find a way to do it like confidence = p[labels]
        # p = p / np.sum(p)
        # confidence = np.max(p, axis=0)
        return labels


    def block_cv_accuracy(self):
        return float(np.sum(self.predict() == self.test_labels))/len(self.test_labels)
        
    def predict_one(self, v):
        prediction_base = np.concatenate((np.eye(self.n_labels, dtype=float), np.tile(v[self.n_labels:].astype(float),(self.n_labels,1))), axis=1)
        p = np.prod(1 + np.exp(self.hbias + np.dot(prediction_base,self.W)), axis=1) * np.exp(self.vbias[:self.n_labels])
        return np.argmax(p)
        
    def cv_accuracy(self, test_set):
        count = 0
        for i in range(len(test_set)):
            count += (np.argmax(test_set[i,:self.n_labels]) == self.predict_one(test_set[i,:]))
        return float(count)/len(test_set)

    def update(self, batch, persistent=False, lr=0.1, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: False for CD, true for PCD.

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        """
        for i in range(len(batch)):
            v0 = batch[i,:]
            # decide how to initialize persistent chain:
            # for CD, we use the sample
            # for PCD, we initialize from the old state of the chain
            if self.persistent is None:
                vk = v0
            else:
                vk = self.persistent[i,:]
                
            # Gibbs sampling
            for i in range(k):
                vk = self.gibbs_vhv(vk)
    
            # determine gradients on RBM parameters
            phv0 = expit(np.dot(v0, self.W) + self.hbias) 
            phvk = expit(np.dot(vk, self.W) + self.hbias) 
            self.W += lr*(np.outer(v0, phv0) - np.outer(vk, phvk))
            self.vbias += lr*(v0 - vk)
            self.hbias += lr*(phv0 - phvk)


        if persistent:
            if self.persistent == None:
                self.persistent = np.zeros((self.batch_size, self.n_visible), dtype=float)
            # Note that this works only if persistent is a shared variable
            self.persistent[i,:] = vk


        