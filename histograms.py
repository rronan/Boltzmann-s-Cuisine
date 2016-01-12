# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:52:20 2016

@author: Ronan
"""

import numpy, pygal
from pygal.style import Style

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
    

# This script needs one object in the environment: le

data = numpy.load('train_data.npy')
freq = numpy.mean(data[:,:20], axis=0)
labels = le.inverse_transform(numpy.array(range(20)))

custom_style = Style(colors = ['#D0D0E0'])
custom_style.label_font_size = 17
custom_style.background = '#FFFFFF'
config = pygal.Config()
config.x_label_rotation = 30
hist = pygal.Bar(config, style = custom_style)
hist.add('cuisine', freq)
hist.x_labels = labels
hist.show_legend = False
hist.render_to_file('_histogram.svg')