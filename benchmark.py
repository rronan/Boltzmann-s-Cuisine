# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:45:45 2015

@author: navrug
"""

import timeit
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

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
score = rf.score(np_test_set[:,20:], np.argmax(np_test_set[:,:20], axis=1))

print score,"in",timeit.default_timer() - start_time, "seconds."