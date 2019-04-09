#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:40:43 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from tqdm import tqdm

import utils
utils.start(__file__)
#==============================================================================

# parameters

params = {
    'bagging_freq': 5,
    'bagging_fraction': 1.0,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.05,
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

NFOLD = 5

NROUND = 150


SEED = 1
np.random.seed(SEED)

search_range = (60, 80)

AUC_bench1 = 0.9260676268235818
AUC_bench2 = 0.9261687712753559


var_len = 200

reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 24, 25, 26,
                27, 29, 32, 35, 37, 40, 41, 47, 48, 49, 51, 52, 53, 55, 60, 61,
                62, 65, 66, 67, 69, 70, 71, 74, 78, 79, 82, 84, 89, 90, 91, 94,
                95, 96, 97, 99, 103, 105, 106, 110, 111, 112, 118, 119, 125, 128,
                130, 133, 134, 135, 137, 138, 140, 144, 145, 147, 151, 155, 157,
                159, 161, 162, 163, 164, 167, 168, 170, 171, 173, 175, 176, 179,
                180, 181, 184, 185, 187, 189, 190, 191, 195, 196, 199,
                ]

# =============================================================================
# main
# =============================================================================

for ayasii in range(search_range[0], search_range[1]):
    print(ayasii)
    # =============================================================================
    # load
    # =============================================================================
    
    train = pd.read_csv("../input/train.csv.zip")
    test  = pd.read_csv("../input/test.csv.zip").drop(np.load('../data/fake_index.npy'))
    
    X_train = train.iloc[:, 2:].values
    y_train = train.target.values
    
    X_test = test.iloc[:, 1:].values
    
    X = np.concatenate([X_train, X_test], axis=0)
    del X_train, X_test; gc.collect()
    
    
    for j in reverse_list:
        X[:, j] *= -1
    
    X[:, ayasii] *= -1
    
    
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # count encoding
    X_cnt = np.zeros((len(X), var_len * 4))
    
    for j in tqdm(range(var_len)):
        for i in range(1, 4):
            x = np.round(X[:, j], i+1)
            dic = pd.value_counts(x).to_dict()
            X_cnt[:, i+j*4] = pd.Series(x).map(dic)
        x = X[:, j]
        dic = pd.value_counts(x).to_dict()
        X_cnt[:, j*4] = pd.Series(x).map(dic)
    
    # raw + count feature
    X_raw = X.copy() # rename for readable
    del X; gc.collect()
    
    X = np.zeros((len(X_raw), var_len * 5))
    for j in tqdm(range(var_len)):
        X[:, 5*j+1:5*j+5] = X_cnt[:, 4*j:4*j+4]
        X[:, 5*j] = X_raw[:, j]
    
    # treat each var as same
    X_train_concat = np.concatenate([
        np.concatenate([
            X[:200000, 5*cnum:5*cnum+5], 
            np.ones((len(y_train), 1)).astype("int")*cnum
        ], axis=1) for cnum in range(var_len)], axis=0)
    y_train_concat = np.concatenate([y_train for cnum in range(var_len)], axis=0)
    
    # =============================================================================
    # train
    # =============================================================================
    
    skf = StratifiedKFold(n_splits=NFOLD)
    skf.get_n_splits(X[:200000, :], y_train)
    
    train_idx_list = []
    valid_idx_list = []
    for train_index, test_index in skf.split(X[:200000, :], y_train):
        train_idx_list.append(train_index)
        valid_idx_list.append(test_index)
    
    models = []
    oof = np.zeros((200000, var_len))
    p_test_all = np.zeros((100000, var_len, NFOLD))
    
    for i in range(NFOLD):
        
        print(f'building {i}...')
        
        train_idx = train_idx_list[i]
        valid_idx = valid_idx_list[i]
        
        # train
        X_train_cv = np.concatenate([
            np.concatenate([
                X[train_idx, 5*cnum:5*cnum+5], 
                np.ones((train_idx.shape[0], 1)).astype("int")*cnum
            ], axis=1) for cnum in range(var_len)], axis=0
        )
        y_train_cv = np.concatenate([y_train[train_idx] for cnum in range(var_len)], axis=0)
        
        # valid
        X_valid = np.concatenate([
            np.concatenate([
                X[valid_idx, 5*cnum:5*cnum+5], 
                np.ones((valid_idx.shape[0], 1)).astype("int")*cnum
            ], axis=1) for cnum in range(var_len)], axis=0
        )
        
        # test
        X_test = np.concatenate([
            np.concatenate([
                X[200000:, 5*cnum:5*cnum+5], 
                np.ones((100000, 1)).astype("int")*cnum
            ], axis=1) for cnum in range(var_len)], axis=0
        )
        
        dtrain = lgb.Dataset(
            X_train_cv, y_train_cv, 
            feature_name=['value', 'count_org', 'count_2', 'count_3', 'count_4', 'varnum'], 
            categorical_feature=['varnum'], free_raw_data=False
        )
        model = lgb.train(params, train_set=dtrain, num_boost_round=NROUND, verbose_eval=100)
        l = valid_idx.shape[0]
        
        p_valid = model.predict(X_valid)
        p_test  = model.predict(X_test)
        for j in range(var_len):
            oof[valid_idx, j]     = p_valid[j*l:(j+1)*l]
            p_test_all[:, j, i] = p_test[j*100000:(j+1)*100000]
        
        models.append(model)
    
    auc = roc_auc_score(y_train, (9 * oof / (1 - oof)).prod(axis=1))
    utils.send_line(f'{ayasii} AUC(all var): {auc}, {AUC_bench1 - auc}')
    
    l = y_train.shape[0]
    oof_odds = np.ones(l) * 1 / 9
    for j in range(var_len):
        if roc_auc_score(y_train, oof[:, j]) >= 0.500:
            oof_odds *= (9 * oof[:, j] / (1 - oof[:, j]))
    
    auc = roc_auc_score(y_train, oof_odds)
    utils.send_line(f'{ayasii} AUC(th0.5): {auc}, {AUC_bench2 - auc}')

#==============================================================================
utils.end(__file__)
utils.stop_instance()

