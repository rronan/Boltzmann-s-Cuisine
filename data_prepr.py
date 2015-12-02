# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:40:48 2015

@author: Ronan
"""

import pandas as pd
import numpy as np

data = pd.read_json(path_or_buf='train.json', 
                    orient=None, 
                    typ='frame', 
                    dtype=True, 
                    convert_axes=True, 
                    convert_dates=True, 
                    keep_default_dates=True, 
                    numpy=False, 
                    precise_float=True, 
                    date_unit=None)

