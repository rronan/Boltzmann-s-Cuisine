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
#        to_keep.append(j-20)
        to_keep.append(j)
        
print "small", small
print "sum", sum(small)
        
test = np.load('test_data.npy')

reduced = test[:,to_keep[20:]]

np.save("test_data_reduced", reduced)