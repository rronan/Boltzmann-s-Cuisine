# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:45:45 2015

@author: navrug
"""

import timeit
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM

data = np.load('train_data.npy')

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

#%%
start_time = timeit.default_timer()


rf = RandomForestClassifier(n_jobs=-1, max_depth=100, n_estimators=200)
rf.fit(np_train_set[:,20:], np.argmax(np_train_set[:,:20], axis=1))
scorerf = rf.score(np_test_set[:,20:], np.argmax(np_test_set[:,:20], axis=1))

print "RandomForest: ", scorerf,"in",timeit.default_timer() - start_time, "seconds."

#%%
start_time = timeit.default_timer()


lr = LogisticRegression(0.1, solver = 'lbfgs', multi_class='multinomial')
lr.fit(np_train_set[:,20:], np.argmax(np_train_set[:,:20], axis=1))
scorelr = lr.score(np_test_set[:,20:], np.argmax(np_test_set[:,:20], axis=1))

print "LogisticRegression: ", scorelr,"in",timeit.default_timer() - start_time, "seconds."

#%%
start_time = timeit.default_timer()

skrbm = BernoulliRBM(batch_size=20, learning_rate=0.01, 
                     n_components=2000, n_iter=50,
                     random_state=None, verbose=1)
skrbm.fit(np_train_set[:,20:], np.argmax(np_train_set[:,:20], axis=1))
scoreskrbm = skrbm.score(np_test_set[:,20:], np.argmax(np_test_set[:,:20], axis=1))

print "BernoulliRBM: ", scoreskrbm,"in",timeit.default_timer() - start_time, "seconds."