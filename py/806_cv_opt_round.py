#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:15:46 2019

@author: Kazuki

find optimal round

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
np.random.seed(SEED)

NFOLD = 10

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
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.9,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.005,
    'max_depth': -1,
    'metric':'binary_logloss',
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 4,
    'num_threads': 32,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1,
    }


NROUND = 99999
ESR = 300
VERBOSE_EVAL = 100

USE_PREF = [
        'f001',
        'f002',
        'f003',
#        'f004',
#        'f005',
        ]

var_names = [f'var_{i:03}' for i in range(200)]

# =============================================================================
# load
# =============================================================================

y_train = utils.load_target()['target']

def load(var):
    
    files_tr = sorted(glob(f'../data/{var}/train_f*.pkl'))
    
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
    
    if X_train.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X_train.shape {X_train.shape}')
    
    gc.collect()
    
    return X_train

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

# =============================================================================
# cv
# =============================================================================

result_all = []

for var in var_names:
    
    X_train_all = load(var)
    auc_best = 0
    round_best = None
    oof_best = None
    result = []
    
    for round_ in [4,3,2,1,0]:
        
        col_drop = get_drop_roundfeature(X_train_all.columns, round_)
        
        X_train = X_train_all.drop(col_drop, axis=1)
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
#                                 feval=ex.eval_auc,
                                 early_stopping_rounds=ESR, 
                                 verbose_eval=VERBOSE_EVAL,
                                 seed=SEED+i)
            
            y_pred = ex.eval_oob(X_train, y_train.values, models, SEED+i, 
                                 stratified=True, shuffle=True)
            y_preds.append(y_pred)
            
            model_all += models
            nround_mean += len(ret['binary_logloss-mean'])
            loss_list.append( ret['binary_logloss-mean'][-1] )
        
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
        
        auc = round(roc_auc_score(y_train, oof), 5)
        
        if auc_best < auc:
            auc_best = auc
            round_best = round_
            oof_best = oof
        
        utils.send_line(f'oof AUC({var, round_}): {auc}')
        result.append(auc)
    
    
    result_all.append(result)
        
    oof_best = pd.DataFrame(oof_best, columns=['oof'])
    oof_best.to_pickle(f'../data/806/oof_{__file__}_{var}_r{round_best}.pkl')
    
#    imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)
#    utils.savefig_imp(imp, f'LOG/imp_{__file__}.png', x='total')

result_all = pd.DataFrame(result_all, 
                          columns=['r4', 'r3', 'r2', 'r1', 'r0'],
                          index=var_names)
result_all.to_csv(f'LOG/auc_{__file__}.csv', index=False)

#==============================================================================
utils.end(__file__)
#utils.stop_instance()
