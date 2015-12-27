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

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        
        # get a sample of the hiddens given their activation

        


    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        
        # get a sample of the visible given their activation


    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''


    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''



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



    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
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
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # perform actual negative phase



        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain


        # We must not compute the gradient through the gibbs sampling,
        # only the RBM parameters.
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])


        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
        # pseudo-likelihood
        monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        return monitoring_cost


    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""



    def get_cv_error(self):

        