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
                          
train_labels = raw_train_data.cuisine
train_id = raw_train_data.id                 

all_ingredients = np.unique(np.hstack(raw_train_data.ingredients))

# TODO: REDUCE DIMENSION VIA CLUSTERING

train_data = pd.DataFrame(data = False, 
                          index = raw_train_data.index, 
                          columns = all_ingredients)

for i in raw_train_data.index:
    for word in raw_train_data.ingredients[i]:
        train_data.ix[i, word] = True
 
np.save('train_labels', train_labels.as_matrix())
np.save('train_id', train_id.as_matrix())
np.save('train_data', train_data.as_matrix())     
#big_matrix.to_csv('big_matrix', encoding = 'utf-8')

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



    