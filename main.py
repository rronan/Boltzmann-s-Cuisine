# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 16:03:09 2015

@author: Gurvan
"""

import timeit

import numpy as np
import sys 
from RBM import SupervisedRBM
from preprocessing import read_data
from sklearn.preprocessing import LabelEncoder
from preprocessing import create_submission

import random

np.random.seed(seed=0)
 
 
learning_rate=0.01
training_epochs=50
batch_size=20
n_chains=20
output_folder='rbm_plots'
n_hidden=2000
dropout_rate=0.5
k=20
do_report = True
load_saved = True

# Create a report to be saved at the end of execution (when running on the 
# remote server)

if load_saved:
    report = np.load("report.npy").item()
elif do_report:
    report = {"learning_rate":learning_rate,
              "training_epochs":training_epochs,
              "batch_size":batch_size,
              "n_chains":n_chains,
              "output_folder":'rbm_plots',
              "n_hidden":n_hidden,
              "dropout_rate":dropout_rate,
              "k":k,
              "costs":np.zeros(training_epochs),
              "accuracy":np.zeros(training_epochs),
              "pretraining_time":0}

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
del data

# compute number of minibatches for training, validation and testing
batches = [np_train_set[i:i + batch_size,:] \
    for i in range(0, np_train_set.shape[0], batch_size)]

rng = np.random.RandomState(123)
# construct the RBM class
rbm = SupervisedRBM(n_visible=n_visible,
          n_labels=n_labels,
          n_hidden=n_hidden, 
          dropout_rate=dropout_rate,
          batch_size=batch_size,
          np_rng=rng)
          
if load_saved:
    rbm.W = report["W"]
    rbm.hbias = report["hbias"]
    rbm.vbias = report["vbias"]
          
#%%============================================================================
# Training the RBM
#==============================================================================


start_time = timeit.default_timer()
accuracies = []
for epoch in range(30,training_epochs):
    epoch_time = timeit.default_timer()
    mean_cost = []
    for batch_index, batch in enumerate(batches):
        rbm.update(batch, persistent=True, k=k)
        sys.stdout.write("\rEpoch advancement: %d%%" % (100*float(batch_index)/len(batches)))
        sys.stdout.flush() 
    if do_report:
        report["W"] = rbm.W
        report["hbias"] = rbm.hbias
        report["vbias"] = rbm.vbias
        np.save('report', report)
    sys.stdout.write("\rEvaluating accuracy...")
    cv_time = timeit.default_timer()
    acc = rbm.cv_accuracy(np_test_set)
    accuracies.append(acc)
    sys.stdout.write('\rEpoch %i took %f minutes, accuracy (computed in %f minutes) is %f.\n'
        % (epoch, ((cv_time-epoch_time) / 60.), ((timeit.default_timer()-cv_time) / 60.), acc))
    if do_report:
        report["costs"][epoch] = np.mean(mean_cost)
        report["accuracy"][epoch] = acc
        
end_time = timeit.default_timer()
pretraining_time = (end_time - start_time)
print ('Training took %f minutes' % (pretraining_time / 60.))

if do_report:
    report["W"] = rbm.W
    report["hbias"] = rbm.hbias
    report["vbias"] = rbm.vbias
    np.save('report', report)

#%%============================================================================
# Make a prediction
#==============================================================================

test_data = np.load('test_data.npy')
test_data = np.concatenate((np.zeros((len(test_data),20)), test_data), axis=1)
y_pred = np.zeros(len(test_data))
for i in xrange(len(y_pred)):
    sys.stdout.write("\rPrediction advancement: %d%%" % (100*float(i)/len(y_pred)))
    sys.stdout.flush() 
    y_pred[i] = rbm.predict_one(test_data[i,:])
train_ids, train_cuisines, train_ingredients = read_data('train.json')
test_ids, test_cuisines, test_ingredients = read_data('test.json')
del train_ids, train_ingredients, test_cuisines, test_ingredients
le = LabelEncoder()
le.fit(train_cuisines)
pred = le.inverse_transform(y_pred.astyp('int'))
create_submission(test_ids, pred)


#%%============================================================================
# Sampling from the RBM
#==============================================================================
from preprocessing import read_data, make_lowercase, remove_numbers, remove_special_chars, remove_extra_whitespace, remove_units, stem_words


train_ids, train_cuisines, train_ingredients = read_data('train.json')
train_ingredients = make_lowercase(train_ingredients)
train_ingredients = remove_numbers(train_ingredients)
train_ingredients = remove_special_chars(train_ingredients)
train_ingredients = remove_extra_whitespace(train_ingredients)
train_ingredients = remove_units(train_ingredients)
train_ingredients = stem_words(train_ingredients)
uniques = np.array(list(set([item for sublist in train_ingredients for item in sublist])))

#%%
rbm.generate(uniques, 5, 1000, obj_ingr=5, n_chains=10)