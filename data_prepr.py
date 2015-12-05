# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:40:48 2015

@author: Ronan
"""

import pandas as pd
import numpy as np

raw_train_data = pd.read_json(path_or_buf='train.json', 
                              orient=None, 
                              typ='frame', 
                              dtype=True, 
                              convert_axes=True, 
                              convert_dates=True, 
                              keep_default_dates=True, 
                              numpy=False, 
                              precise_float=True, 
                              date_unit=None)
                          
train_id = raw_train_data.id                 

all_ingredients = np.unique(np.hstack(raw_train_data.ingredients))
all_labels = np.unique(raw_train_data.cuisine)

# TODO: REDUCE DIMENSION VIA CLUSTERING

# TODO: STEMMING

# TODO: REMOVE figures

train_data_full = pd.DataFrame(data = False, 
                          index = raw_train_data.index, 
                          columns = np.hstack([all_labels, all_ingredients]))

for i in raw_train_data.index:
    train_data_full.ix[i, raw_train_data.cuisine[i]] = True
    for word in raw_train_data.ingredients[i]:
        train_data_full.ix[i, word] = True
 
# REMOVE ingredients present in less than r% of the recipes       
r = 0.01        
idx = train_data_full.sum(axis = 0) > (train_data_full.shape[0] * r)
train_data = train_data_full[np.where(idx)[0]]

np.save('train_id', train_id.as_matrix())
np.save('train_data', train_data.as_matrix())

###############################################################################

raw_test_data = pd.read_json(path_or_buf='test.json', 
                              orient=None, 
                              typ='frame', 
                              dtype=True, 
                              convert_axes=True, 
                              convert_dates=True, 
                              keep_default_dates=True, 
                              numpy=False, 
                              precise_float=True, 
                              date_unit=None)
                          
test_id = raw_test_data.id                 

# APPLY SAME CLUSTERING TO TEST DATA

test_data = pd.DataFrame(data = False, 
                         index = raw_test_data.index, 
                         columns = all_ingredients)

count = 0
for i in raw_test_data.index:
    for word in raw_test_data.ingredients[i]:
        try:
            test_data.ix[i, word] = True
        except KeyError:
            count += 1
            print word + 'not in train_data';
            
np.save('test_id', train_id.as_matrix())
np.save('test_data', train_data.as_matrix())
