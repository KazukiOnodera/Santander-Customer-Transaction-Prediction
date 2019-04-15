#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:37:01 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc, os

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from tqdm import tqdm

import sys
#==============================================================================

# parameters

params = {
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

NFOLD = 10

NROUND = 1600


SEED = int(sys.argv[1])
np.random.seed(SEED)

SUBMIT_FILE_PATH = f'../output/0410-1_seed{SEED}.csv.gz'

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
train = pd.read_csv("../input/train.csv.zip")
test  = pd.read_csv("../input/test.csv.zip").drop(np.load('../data/fake_index.npy'))

X_train = train.iloc[:, 2:].values
y_train = train.target.values

X_test = test.iloc[:, 1:].values

X = np.concatenate([X_train, X_test], axis=0)
del X_train, X_test; gc.collect()

reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 24, 25, 26,
                27, 29, 32, 35, 37, 40, 41, 47, 48, 49, 51, 52, 53, 55, 60, 61,
                62, 65, 66, 67, 69, 70, 71, 74, 78, 79, 82, 84, 89, 90, 91, 94,
                95, 96, 97, 99, 103, 105, 106, 110, 111, 112, 118, 119, 125, 128,
                130, 133, 134, 135, 137, 138, 140, 144, 145, 147, 151, 155, 157,
                159, 161, 162, 163, 164, 167, 168, 170, 171, 173, 175, 176, 179,
                180, 181, 184, 185, 187, 189, 190, 191, 195, 196, 199,
                
                ]

for j in reverse_list:
    X[:, j] *= -1


# drop
X = np.delete(X, drop_vars, 1)


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
# stratified
# =============================================================================
train_group = np.arange(len(X_train_concat))%200000

id_y = pd.DataFrame(zip(train_group, y_train_concat), 
                    columns=['id', 'y'])

id_y_uq = id_y.drop_duplicates('id').reset_index(drop=True)

def stratified(nfold=5):
    
    id_y_uq0 = id_y_uq[id_y_uq.y==0].sample(frac=1)
    id_y_uq1 = id_y_uq[id_y_uq.y==1].sample(frac=1)
    
    id_y_uq0['g'] = [i%nfold for i in range(len(id_y_uq0))]
    id_y_uq1['g'] = [i%nfold for i in range(len(id_y_uq1))]
    id_y_uq_ = pd.concat([id_y_uq0, id_y_uq1])
    
    id_y_ = pd.merge(id_y[['id']], id_y_uq_, how='left', on='id')
    
    train_idx_list = []
    valid_idx_list = []
    for i in range(nfold):
        train_idx = id_y_[id_y_.g!=i].index
        train_idx_list.append(train_idx)
        valid_idx = id_y_[id_y_.g==i].index
        valid_idx_list.append(valid_idx)
    
    return train_idx_list, valid_idx_list

train_idx_list, valid_idx_list = stratified(NFOLD)

# =============================================================================
# cv
# =============================================================================

#dtrain = lgb.Dataset(X_train_concat, y_train_concat, 
#    feature_name=['value', 'count_org', 'count_2', 'count_3', 'count_4', 'varnum'], 
#    categorical_feature=['varnum'], free_raw_data=False)
#
#del X_train_concat; gc.collect()

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


# =============================================================================
# train
# =============================================================================

models = []
oof = np.zeros(len(id_y))
p_test_all = np.zeros((100000, var_len, NFOLD))
id_y['var'] = np.concatenate([np.ones(200000)*i for i in range(var_len)])

for i in range(NFOLD):
    
    print(f'building {i}...')
    
    train_idx = train_idx_list[i]
    valid_idx = valid_idx_list[i]
    
    # train
    X_train_cv = X_train_concat[train_idx]
    y_train_cv = y_train_concat[train_idx]
    
    # valid
    X_valid = X_train_concat[valid_idx]
    
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
        oof[valid_idx] = p_valid
        p_test_all[:, j, i] = p_test[j*100000:(j+1)*100000]
    
    models.append(model)

id_y['pred'] = oof
oof = pd.pivot_table(id_y, index='id', columns='var', values='pred').values

auc = f'seed{SEED} AUC(all var): {roc_auc_score(y_train, (9 * oof / (1 - oof)).prod(axis=1))}'
print(auc)

l = y_train.shape[0]
oof_odds = np.ones(l) * 1 / 9
for j in range(var_len):
    if roc_auc_score(y_train, oof[:, j]) >= 0.500:
        oof_odds *= (9 * oof[:, j] / (1 - oof[:, j]))

auc = f'seed{SEED} AUC(th0.5): {roc_auc_score(y_train, oof_odds)}'
print(auc)

# =============================================================================
# test
# =============================================================================

p_test_mean = p_test_all.mean(axis=2)

p_test_odds = np.ones(100000) * 1 / 9
for j in range(var_len):
    if roc_auc_score(y_train, oof[:, j]) >= 0.500:
        p_test_odds *= (9 * p_test_mean[:, j] / (1 - p_test_mean[:, j]))

p_test_odds = p_test_odds / (1 + p_test_odds)

sub1 = pd.read_csv("../input/sample_submission.csv.zip")
sub2 = pd.DataFrame({"ID_code":test.ID_code.values , "target":p_test_odds})
sub = pd.merge(sub1[["ID_code"]], sub2, how="left").fillna(0)


# save
sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')


#==============================================================================


