#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:19:03 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox

dist1 = pd.Series(np.random.exponential(scale=10, size=1000))

dist1.hist(bins=30)

nom = pd.Series(boxcox(dist1)[0])


nom.hist(bins=30)

