# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:17:06 2015

@author: Ronan
"""

import numpy

best_params = numpy.load('reports/best_params.npy')
hyper_scores = numpy.load('reports/hyper_scores.npy')
#print best_params

report = numpy.load('reports/report_6.npy')
print report
