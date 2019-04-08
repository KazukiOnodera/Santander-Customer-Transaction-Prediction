#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:46:56 2019

@author: Kazuki
"""

#import sys
#import os
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#%matplotlib inline
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from tqdm import tqdm

import utils
utils.start(__file__)
#==============================================================================

# =============================================================================
# parameters
# =============================================================================

SUBMIT_FILE_PATH = '../output/0408-1.csv.gz'

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


SEED = 4081

# =============================================================================
# drop vars
# =============================================================================

drop_vars = [7,
            10,
            17,
            27,
            29,
            30,
            38,
            41,
            46,
            96,
            100,
            103,
            126,
            158,
            185]

var_len = 200 - len(drop_vars)


# =============================================================================
# load
# =============================================================================
train_df = pd.read_csv("../input/train.csv.zip")
test_df  = pd.read_csv("../input/test.csv.zip").drop(np.load('../data/fake_index.npy'))
train_x = train_df.iloc[:, 2:].values
test_x = test_df.iloc[:, 1:].values
train_y = train_df.target.values

y_train = train_df.target.values

train_test_x_org = np.concatenate([train_x, test_x], axis=0)

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
for j in reverse_list:
    train_test_x_org[:, j] *= -1

# drop
train_test_x_org = np.delete(train_test_x_org, drop_vars, 1)


# scaling
scaler = StandardScaler()
train_test_x = scaler.fit_transform(train_test_x_org)

train_test_x_cnt = np.zeros((train_test_x.shape[0], var_len * 4))

for j in tqdm(range(var_len)):
    for i in range(1, 4):
        x = np.round(train_test_x[:, j], i+1)
        dic = pd.value_counts(x).to_dict()
        train_test_x_cnt[:, i+j*4] = pd.Series(x).map(dic)
    x = train_test_x[:, j]
    dic = pd.value_counts(x).to_dict()
    train_test_x_cnt[:, j*4] = pd.Series(x).map(dic)
    
train_test_x2 = np.zeros((train_test_x.shape[0], var_len * 5))
for j in tqdm(range(var_len)):
    train_test_x2[:, 5*j+1:5*j+5] = train_test_x_cnt[:, 4*j:4*j+4]
    train_test_x2[:, 5*j] = train_test_x[:, j]

# =============================================================================
# train
# =============================================================================

tratest_X = np.concatenate([
    np.concatenate([
        train_test_x2[:200000, 5*cnum:5*cnum+5], 
        np.ones((y_train.shape[0], 1)).astype("int")*cnum
    ], axis=1) for cnum in range(var_len)], axis=0
)
tratest_y = np.concatenate([y_train for cnum in range(var_len)], axis=0)
tratest_dset = lgb.Dataset(
    tratest_X, tratest_y, 
    feature_name=['value', 'count_org', 'count_2', 'count_3', 'count_4', 'varnum'], 
    categorical_feature=['varnum'], free_raw_data=False)

del tratest_X

# cv
#ret, models, = lgb.cv(params, tratest_dset, early_stopping_rounds=200, nfold=NFOLD, 
#                      num_boost_round=10000, verbose_eval=100)
"""
[100]	cv_agg's binary_logloss: 0.484652 + 2.26578e-06
[200]	cv_agg's binary_logloss: 0.395391 + 8.73085e-06
[300]	cv_agg's binary_logloss: 0.355365 + 6.98843e-06
[400]	cv_agg's binary_logloss: 0.33769 + 8.28864e-06
[500]	cv_agg's binary_logloss: 0.33015 + 1.02757e-05
[600]	cv_agg's binary_logloss: 0.327054 + 1.20748e-05
[700]	cv_agg's binary_logloss: 0.325823 + 1.33061e-05
[800]	cv_agg's binary_logloss: 0.325346 + 1.3853e-05
[900]	cv_agg's binary_logloss: 0.325164 + 1.41864e-05
[1000]	cv_agg's binary_logloss: 0.325096 + 1.40618e-05
[1100]	cv_agg's binary_logloss: 0.32507 + 1.43292e-05
[1200]	cv_agg's binary_logloss: 0.325061 + 1.4517e-05
[1300]	cv_agg's binary_logloss: 0.325058 + 1.46426e-05
[1400]	cv_agg's binary_logloss: 0.325057 + 1.47049e-05
[1500]	cv_agg's binary_logloss: 0.325057 + 1.47674e-05
[1600]	cv_agg's binary_logloss: 0.325057 + 1.48108e-05
"""

skf = StratifiedKFold(n_splits=NFOLD)
skf.get_n_splits(train_test_x2[:200000, :], train_y)

train_idx_list = []
valid_idx_list = []
for train_index, test_index in skf.split(train_test_x2[:200000, :], train_y):
    train_idx_list.append(train_index)
    valid_idx_list.append(test_index)

clf_list = []
oof_pred = np.zeros((200000, var_len))
tes_pred = np.zeros((100000, var_len, NFOLD))

for i in range(NFOLD):
    
    trn_idx = train_idx_list[i]
    val_idx = valid_idx_list[i]

    tra_X = np.concatenate([
        np.concatenate([
            train_test_x2[trn_idx, 5*cnum:5*cnum+5], 
            np.ones((trn_idx.shape[0], 1)).astype("int")*cnum
        ], axis=1) for cnum in range(var_len)], axis=0
    )
    tra_y = np.concatenate([y_train[trn_idx] for cnum in range(var_len)], axis=0)

    val_X = np.concatenate([
        np.concatenate([
            train_test_x2[val_idx, 5*cnum:5*cnum+5], 
            np.ones((val_idx.shape[0], 1)).astype("int")*cnum
        ], axis=1) for cnum in range(var_len)], axis=0
    )
    tes_X = np.concatenate([
        np.concatenate([
            train_test_x2[200000:, 5*cnum:5*cnum+5], 
            np.ones((100000, 1)).astype("int")*cnum
        ], axis=1) for cnum in range(var_len)], axis=0
    )
    
    train_dset = lgb.Dataset(
        tra_X, tra_y, 
        feature_name=['value', 'count_org', 'count_2', 'count_3', 'count_4', 'varnum'], 
        categorical_feature=['varnum'], free_raw_data=False
    )
    clf = lgb.train(params, train_set=train_dset, num_boost_round=NROUND, verbose_eval=100)
    l = val_idx.shape[0]
    
    pred_valid = clf.predict(val_X)
    pred_tes = clf.predict(tes_X)
    for j in range(var_len):
        oof_pred[val_idx, j] = pred_valid[j*l:(j+1)*l]
        tes_pred[:, j, i] = pred_tes[j*100000:(j+1)*100000]
    
    clf_list.append(clf)
    print("i = ", i)

print('AUC(all var):', roc_auc_score(y_train, (9 * oof_pred / (1 - oof_pred)).prod(axis=1)))

l = y_train.shape[0]
pred_valid_p = np.ones(l) * 1 / 9
for j in range(var_len):
    if roc_auc_score(y_train, oof_pred[:, j]) >= 0.500:
        pred_valid_p *= (9 * oof_pred[:, j] / (1 - oof_pred[:, j]))

print('AUC(th0.5):', roc_auc_score(y_train, pred_valid_p))

# save raw pred
np.save('../data/{__file__}_oof_pred', oof_pred)
np.save('../data/{__file__}_tes_pred', tes_pred)

# =============================================================================
# test
# =============================================================================

pred_tes_m = tes_pred.mean(axis=2)

pred_test_p = np.ones(100000) * 1 / 9
for j in range(var_len):
    if roc_auc_score(y_train, oof_pred[:, j]) >= 0.500:
        pred_test_p *= (9 * pred_tes_m[:, j] / (1 - pred_tes_m[:, j]))

pred_test_pp = pred_test_p / (1 + pred_test_p)

samp = pd.read_csv("../input/sample_submission.csv.zip")
test_df_samp = pd.DataFrame({"ID_code":test_df.ID_code.values , "target":pred_test_pp})
sub_df_ = pd.merge(samp[["ID_code"]], test_df_samp, how="left").fillna(0)


# save
sub_df_.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

print(sub_df_.target.describe())


#==============================================================================
utils.end(__file__)
#utils.stop_instance()







