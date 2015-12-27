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

import numpy

from scipy.special import expit


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        n_visible,
        n_labels,
        n_hidden,
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

        if np_rng is None:
            # create a number generator
            np_rng = numpy.random.RandomState(0)

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
            initial_W = numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=float
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=float
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=float
                ),
                name='vbias',
                borrow=True
            )
            
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
        

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''


    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        
        NOTE: I STILL RETURN THE TWO QUANTITIES FOR THE MOMENT

        '''
        pre_sigmoid_activation = numpy.dot(vis, self.W) + self.hbias
        
        return [pre_sigmoid_activation, expit(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = numpy.random.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        
        return [pre_sigmoid_h1, h1_mean, h1_sample]        


    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        
        NOTE: I STILL RETURN THE TWO QUANTITIES FOR THE MOMENT

        '''
        pre_sigmoid_activation = numpy.dot(hid, self.W.T) + self.vbias
        
        return [pre_sigmoid_activation, expit(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = numpy.random.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        
        return [pre_sigmoid_v1, v1_mean, v1_sample]


    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]



    def compute_prob(self, unlabelled):
        ''' This function makes a prediction for an unlabelled sample,
        this is done by computing Z.P(v_unlablled,label) which is proportional
        to P(label|v_unlabelled).'''

        
                
    def predict(self, unlabelled):
        ''' This function makes a prediction for an unlabelled sample,
        this is done by computing Z.P(v_unlablled,label) which is proportional
        to P(label|v_unlabelled).'''
        normed_prob = self.compute_prob(unlabelled)
        labels = T.argmax(normed_prob,axis=1)
        # TODO: find a way to do it like confidence = normed_prob[labels]
        confidence = T.max(normed_prob,axis=1)
        return labels, confidence



    def update(self, batch, persistent=False, lr=0.1, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        
        for i in len(batch):
            v0 = batch[i,:]
            # decide how to initialize persistent chain:
            # for CD, we use the sample
            # for PCD, we initialize from the old state of the chain
            if self.persistent is None:
                vk = v0
            else:
                vk = self.persistent[i,:,:]
                
            # Gibbs sampling
            for i in range(k):
                vk = gibbs_vhv(vk)
    
            # determine gradients on RBM parameters
            phv0 = sigmoid(np.dot(self.W, v0) + self.hbias) 
            phvk = sigmoid(np.dot(self.W, vk) + self.hbias) 
            self.W += lr*(np.outer(v0, phv0) - np.outer(vk, phvk))
            self.vbias += lr*(v0 - vk)
            self.hbias += lr*(phv0 - phv)


        if persistent:
            # Note that this works only if persistent is a shared variable
            self.persistent[i,:,:] = vk


    def get_cv_error(self):

        