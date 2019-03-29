#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:53:37 2019

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

import utils

#utils.start(__file__)
# =============================================================================

#SUBMIT_FILE_PATH = '../output/0328-1.csv.gz'
#
#COMMENT = 'lgb shuffle row'

EXE_SUBMIT = True

NFOLD = 5

LOOP = 1

param = {
         'objective': 'binary',
         'metric': 'None',
         
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 2**6 -1,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }


NROUND = 9999
ESR = 100
VERBOSE_EVAL = 50
SEED = np.random.randint(9999)


# =============================================================================
# load
# =============================================================================
X_train = pd.read_csv('../input/train.csv.zip')

y_train = X_train['target']
X_train = X_train.iloc[:,2:]

X_train_0 = X_train[y_train==0]
X_train_1 = X_train[y_train==1]

def shuffle(df):
    df_ = pd.DataFrame(index=df.index)
    for c in tqdm(df.columns):
        df_[c] = df[c].sample(frac=1).reset_index(drop=True)
    return df_


X_train_ = pd.concat([shuffle(X_train_0), shuffle(X_train_0), shuffle(X_train_0),
                      shuffle(X_train_1), shuffle(X_train_1), shuffle(X_train_1)]).sort_index()
y_train_ = pd.concat([y_train, y_train, y_train]).sort_index()


# =============================================================================
# model
# =============================================================================
dtrain = lgb.Dataset(X_train_, y_train_.values, 
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
#                         folds=group_kfold.split(X_train_, y_train_, group),
                         stratified=True, shuffle=True,
                         feval=ex.eval_auc,
                         early_stopping_rounds=ESR, 
                         verbose_eval=VERBOSE_EVAL,
                         seed=SEED+i)
    
    y_pred = ex.eval_oob(X_train_, y_train_.values, models, SEED+i, 
#                         folds=group_kfold.split(X_train_, y_train_, group),
                         stratified=True, shuffle=True)
    y_preds.append(y_pred)
    
    model_all += models
    nround_mean += len(ret['auc-mean'])
    loss_list.append( ret['auc-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)


## =============================================================================
## test
## =============================================================================
#
#test = pd.read_csv('../input/test.csv.zip')
#
#sub = pd.read_csv('../input/sample_submission.csv.zip')
#
#for model in tqdm(models):
#    sub['target'] += pd.Series(model.predict(test.iloc[:,1:])).rank()
#sub['target'] /= sub['target'].max()
#
#
#
#
#
#
#
## save
#sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')
#
## =============================================================================
## submission
## =============================================================================
#if EXE_SUBMIT:
#    print('submit')
#    utils.submit(SUBMIT_FILE_PATH, COMMENT)
#
#
#
##==============================================================================
#utils.end(__file__)
##utils.stop_instance()


