#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:14:37 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from time import sleep
#sleep(60 * 5)

from sklearn.metrics import roc_auc_score

import utils
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)


NFOLD = 5


param = {
         'objective': 'binary',
         'metric': 'None',
         
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 2**6 -1,
#         'num_leaves': 2**4 -1,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 5,
         'verbose':-1,
         }


NROUND = 99999
ESR = 100
VERBOSE_EVAL = 50

USE_PREF = [
        'f001',
        'f003',
#        'f004',
#        'f005',
#        'f006',
#        'f007',
#        'f008',
#        'f009',
#        'f010',
#        'f011',
#        'f012',
        ]

COL = pd.read_csv('LOG/imp_801_cv.py.csv').feature.tolist()


# =============================================================================
# load
# =============================================================================


files_tr = sorted(glob('../data/train_f*.pkl'))

# USE_PREF
li = []
for i in files_tr:
    for j in USE_PREF:
        if j in i:
            li.append(i)
            break
files_tr = li

[print(i,f) for i,f in enumerate(files_tr)]

X_train = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=30)
               ], axis=1)

y_train = utils.load_target()['target']



if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()


# =============================================================================
# cv
# =============================================================================


model_all = []
nround_mean = 0
loss_list = []

for i in range(100, 900, 100):
    col = COL[:i]
    dtrain = lgb.Dataset(X_train[col], y_train.values, 
                     free_raw_data=False)
    gc.collect()
    
    param['seed'] = np.random.randint(9999)
    
    ret, models = lgb.cv(param, dtrain, NROUND,
                         nfold=NFOLD,
                         stratified=True, shuffle=True,
                         feval=ex.eval_auc,
                         early_stopping_rounds=ESR, 
                         verbose_eval=VERBOSE_EVAL,
                         seed=SEED+i)
    
    p_train = ex.eval_oob(X_train, y_train.values, models, SEED+i, 
                         stratified=True, shuffle=True)
    
    model_all += models
    nround_mean += len(ret['auc-mean'])
    loss_list.append( ret['auc-mean'][-1] )
    
    utils.send_line(f'oof AUC({i}): {round(roc_auc_score(y_train, p_train), 5)}')




#==============================================================================
utils.end(__file__)
#utils.stop_instance()

