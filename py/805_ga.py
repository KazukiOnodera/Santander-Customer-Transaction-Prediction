#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:31:05 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import GA

import utils
#utils.start(__file__)


# =============================================================================
# load
# =============================================================================
X = pd.read_pickle('../external/share_904_oof_preds.pkl.gz')
oof_pred_array = X.values

y = utils.load_target()['target']


print(roc_auc_score(y, (9 * oof_pred_array / (1 - oof_pred_array)).prod(axis=1)))

# =============================================================================
# def
# =============================================================================
def myfitness(gtype):
    """
    gtype[:200]: weight
    gtype[200:]: binary(use or not)
    
    """
    
    pred = oof_pred_array * np.array(gtype[:200])
    usecols = [i for i,e in enumerate(gtype[200:]) if e>0]
    pred = pred[:, usecols]
    auc = roc_auc_score(y, (9 * pred / (1 - pred)).prod(axis=1))
    return auc

myfitness(np.r_[np.ones(200), np.ones(200)])



auc_arr = np.array([roc_auc_score(y, oof_pred_array[0:,i]) for i in tqdm(range(200))])


[print(th, myfitness(np.r_[np.ones(200), auc_arr > th])) for th in np.arange(0.495, 0.510, 0.001)]

init_gtype = list(np.r_[np.ones(200), auc_arr > 0.505])

# =============================================================================
# GA
# =============================================================================

THRESHOLD = [{'min':0, 'max':1, 'type':float, 'round':3} for i in range(200)]
THRESHOLD += [{'min':0, 'max':1, 'type':int,} for i in range(200)]

ga = GA.GA(THRESHOLD, 
            generation=100,
            population=10000,
            feval=myfitness, 
            init_gtype=init_gtype,
            maximize=True, 
            is_print=0,
            n_jobs=8,
            to_csv='LOG/ga{i}.csv'
            )


ga.fit()









#utils.end(__file__)

