#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:43:14 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

result_all = pd.read_csv('/Users/Kazuki/Downloads/auc_806_cv_opt_round.py.csv')


var_names = [f'var_{i:03}' for i in range(200)]

result_all.index = var_names

result_all_norm = pd.DataFrame(sc.fit_transform(result_all.T).T, 
                     columns=result_all.columns,
                     index=result_all.index)


sns.heatmap(result_all_norm, cmap='bwr')

