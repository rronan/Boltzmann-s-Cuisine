# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:03:21 2015

@author: navrug
"""

import numpy as np

data = np.load('train_data.npy')

to_keep = []
th = 200
small = np.zeros(th)
for j in range(len(data[0])):
    if np.sum(data[:,j]) < th:
        small[np.sum(data[:,j])] += 1
    else:
        to_keep.append(j)
        
print "small", small
print "sum", sum(small)
        
reduced = data[:,to_keep]

np.save("train_data_reduced", reduced)