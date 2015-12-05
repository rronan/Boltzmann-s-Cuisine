# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:50:30 2015

@author: Ronan
"""

import theano
import numpy

import theano.tensor as T

def load_data(train_set, test_set):

    def shared_dataset(data, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_data = theano.shared(numpy.asarray(data,
                                                  dtype=theano.config.floatX),
                                    borrow=borrow)

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_data

    train_set = shared_dataset(train_set)
    test_set = shared_dataset(test_set)

    rval = [train_set, test_set]
    return rval