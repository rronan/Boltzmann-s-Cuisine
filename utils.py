# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:17:06 2015

@author: Ronan
"""

import numpy

report_folder = "reports_27_12/"

best_params = numpy.load(report_folder + 'best_params.npy')
hyper_scores = numpy.load(report_folder+ 'hyper_scores.npy')
print best_params
print hyper_scores

report = numpy.load(report_folder + 'report_1.npy')
print report
