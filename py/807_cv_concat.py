#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:36:18 2019

@author: Kazuki
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
from sklearn.preprocessing import StandardScaler

import utils
#utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)


NFOLD = 5

LOOP = 2

#param = {
#         'objective': 'binary',
#         'metric': 'None',
#         
#         'learning_rate': 0.01,
#         'max_depth': -1,
#         'num_leaves': 2**6 -1,
##         'num_leaves': 2**4 -1,
#         'max_bin': 255,
#         
#         'min_child_weight': 10,
#         'min_data_in_leaf': 150,
#         'reg_lambda': 0.5,  # L2 regularization term on weights.
#         'reg_alpha': 0.5,  # L1 regularization term on weights.
#         
#         'colsample_bytree': 0.5,
#         'subsample': 0.7,
##         'nthread': 32,
#         'nthread': cpu_count(),
#         'bagging_freq': 5,
#         'verbose':-1,
#         }


# akiyama param
#param = {
#    'bagging_freq': 5,
#    'bagging_fraction': 0.9,
#    'boost_from_average':'false',
#    'boost': 'gbdt',
#    'feature_fraction': 1.0,
#    'learning_rate': 0.005,
#    'max_depth': -1,
#    'metric':'binary_logloss',
#    'min_data_in_leaf': 10,
#    'min_sum_hessian_in_leaf': 10.0,
#    'num_leaves': 4,
#    'num_threads': cpu_count(),
#    'tree_learner': 'serial',
#    'objective': 'binary',
#    'verbosity': -1,
#    }

# harada param
param = {
    'bagging_freq': 5,
    'bagging_fraction': 1.0,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.005,
    'max_depth': -1,
    'metric':'binary_logloss',
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 64,
    'num_threads': cpu_count(),
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
    }

NROUND = 99999
ESR = 100
VERBOSE_EVAL = 50

USE_PREF = [
        'f001',
        'f002',
        'f003',
        ]

USE_ROUND = 4

DROP_VAR  = []

var_names = [f'var_{i:03}' for i in range(200)]
var_names = [var for var in var_names if var not in DROP_VAR]

reverse_list = [0,1,2,3,4,5,6,7,8,11,15,16,18,19,
            22,24,25,26,27,41,29,
            32,35,37,40,48,49,47,
            55,51,52,53,60,61,62,103,65,66,67,69,
            70,71,74,78,79,
            82,84,89,90,91,94,95,96,97,99,
            105,106,110,111,112,118,119,125,128,
            130,133,134,135,137,138,
            140,144,145,147,151,155,157,159,
            161,162,163,164,167,168,
            170,171,173,175,176,179,
            180,181,184,185,187,189,
            190,191,195,196,199]
reverse_list = [f'var_{i:03}' for i in reverse_list]

# =============================================================================
# def
# =============================================================================
scaler = StandardScaler()

def get_drop_roundfeature(col, round_):
    """
    
    if round_ is 3, use by *_r3, drop *_r2~*_r0
    
    """
    
    col_r3 = [c for c in col if c.endswith('_r3')]
    col_r2 = [c for c in col if c.endswith('_r2')]
    col_r1 = [c for c in col if c.endswith('_r1')]
    col_r0 = [c for c in col if c.endswith('_r0')]
    
    if round_ == 4:
        return col_r3 + col_r2 + col_r1 + col_r0
    
    if round_ == 3:
        return col_r2 + col_r1 + col_r0
    
    elif round_ == 2:
        return col_r1 + col_r0
    
    elif round_ == 1:
        return col_r0
    
    elif round_ == 0:
        return []
    
    else:
        raise(round_)
    return

def load(var):
    
    # train
    files = sorted(glob(f'../data/{var}/train_f*.pkl'))
    
    # USE_PREF
    li = []
    for i in files:
        for j in USE_PREF:
            if j in i:
                li.append(i)
                break
    files = li
    
    X_train = pd.concat([
                    pd.read_pickle(f) for f in files
                   ], axis=1)
    
    # test
    files = sorted(glob(f'../data/{var}/test_f*.pkl'))
    
    # USE_PREF
    li = []
    for i in files:
        for j in USE_PREF:
            if j in i:
                li.append(i)
                break
    files = li
    
    X_test = pd.concat([
                    pd.read_pickle(f) for f in files
                   ], axis=1)
    
    X = pd.concat([X_train, X_test], ignore_index=True)
    
    if var in reverse_list:
        X[f'f001_{var}'] *= -1
    
    col = [c.replace('_'+var, '') for c in X.columns]
    
    # scaling
    X = pd.DataFrame(scaler.fit_transform(X), columns=col)
    X['is_test'] = 0
    X.loc[200000:, 'is_test'] = 1
    
    return X

# =============================================================================
# load train and test
# =============================================================================

X = pd.concat([
                load(var) for var in tqdm(var_names, mininterval=30)
               ], ignore_index=True)

col_drop = get_drop_roundfeature(X.columns, USE_ROUND)
X.drop(col_drop, axis=1, inplace=True)


X_train = X[X['is_test']==0].reset_index(drop=True)
X_test = X[X['is_test']==1].reset_index(drop=True)

del X_train['is_test'], X_test['is_test']

y_train = utils.load_target()['target']
y_train = pd.concat([y_train for var in var_names])


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')


if X_test.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_test.columns[X_test.columns.duplicated()] }')
print('no dup :) ')
print(f'X_test.shape {X_test.shape}')

del X, X_test
gc.collect()


# =============================================================================
# cv
# =============================================================================

dtrain = lgb.Dataset(X_train, y_train.values, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
loss_list = []
y_preds = []
for i in range(LOOP):
    gc.collect()
    
    param['seed'] = np.random.randint(9999)
    
    ret, models = lgb.cv(param, dtrain, NROUND,
                         nfold=NFOLD,
                         stratified=True, shuffle=True,
                         feval=ex.eval_auc,
                         early_stopping_rounds=ESR, 
                         verbose_eval=VERBOSE_EVAL,
                         seed=SEED+i)
    
    y_pred = ex.eval_oob(X_train, y_train.values, models, SEED+i, 
                         stratified=True, shuffle=True)
    y_preds.append(y_pred)
    
    model_all += models
    nround_mean += len(ret['auc-mean'])
    loss_list.append( ret['auc-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)


imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)


for i,y_pred in enumerate(y_preds):
    if i==0:
        oof = y_pred
    else:
        oof += y_pred
oof /= len(y_preds)



imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)
pd.DataFrame(oof, columns=['oof']).to_csv(f'../data/oof_{__file__}.csv', index=False)

utils.savefig_imp(imp, f'LOG/imp_{__file__}.png', x='total')


utils.send_line(f'oof AUC: {round(roc_auc_score(y_train, oof), 5)}')

#==============================================================================
utils.end(__file__)
#utils.stop_instance()

