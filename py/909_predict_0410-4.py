#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:51:06 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc, os

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from tqdm import tqdm

import utils
utils.start(__file__)
#==============================================================================

# parameters
SEED = np.random.randint(99999)
np.random.seed(SEED)

SUBMIT_FILE_PATH = f'../output/0410-4_seed{SEED}.csv.gz'


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

NFOLD = 5

NROUND = 1600


var_len = 200

reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 24, 25, 26,
                27, 29, 32, 35, 37, 40, 41, 47, 48, 49, 51, 52, 53, 55, 60, 61,
                62, 65, 66, 67, 69, 70, 71, 74, 78, 79, 82, 84, 89, 90, 91, 94,
                95, 96, 97, 99, 103, 105, 106, 110, 111, 112, 118, 119, 125, 128,
                130, 133, 134, 135, 137, 138, 140, 144, 145, 147, 151, 155, 157,
                159, 161, 162, 163, 164, 167, 168, 170, 171, 173, 175, 176, 179,
                180, 181, 184, 185, 187, 189, 190, 191, 195, 196, 199,
                ]

pseudo_threshold = 1500
best_sub_path = '../output/0410-3.csv.gz'

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

train_group = np.arange(len(X_train_concat))%200000

# =============================================================================
# add test
# =============================================================================
sub = pd.read_csv(best_sub_path).drop(np.load('../data/fake_index.npy')).reset_index(drop=True)
sub.sort_values('target', ascending=0, inplace=True)
pseudo_index = sub.head(pseudo_threshold).index + 200000
#pseudo_label_test = np.array([(i in pseudo_index)*1 for i in range(200000,300000)])

X_test_concat = np.concatenate([
    np.concatenate([
        X[pseudo_index, 5*cnum:5*cnum+5], 
        np.ones((len(pseudo_index), 1)).astype("int")*cnum
    ], axis=1) for cnum in range(var_len)], axis=0)
y_test_concat = np.concatenate([np.ones(len(pseudo_index)) for cnum in range(var_len)], axis=0)

test_group = 200000 + (np.arange(len(X_test_concat)) % pseudo_threshold)

X_train_concat = np.concatenate([X_train_concat, X_test_concat])
y_train_concat = np.concatenate([y_train_concat, y_test_concat])

id_y = pd.DataFrame(zip(np.r_[train_group, test_group], y_train_concat), 
                    columns=['id', 'y'])

id_y_uq = id_y.drop_duplicates('id').reset_index(drop=True)

y_train_pseudo = np.r_[y_train, np.ones(len(pseudo_index))]

# =============================================================================
# train
# =============================================================================

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

train_idx_list, valid_idx_list = stratified( NFOLD)


models = []
#oof = np.zeros((len(id_y_uq), var_len))
oof = np.zeros(len(id_y))
p_test_all = np.zeros((100000, var_len, NFOLD))
id_y['var'] = np.r_[np.concatenate([np.ones(200000)*i for i in range(var_len)]), 
                    np.concatenate([np.ones(len(pseudo_index))*i for i in range(var_len)])]
#id_y['pred'] = 0

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
oof = pd.pivot_table(id_y, index='id', columns='var', values='pred').head(200000).values

auc = roc_auc_score(y_train, (9 * oof / (1 - oof)).prod(axis=1))
utils.send_line(f'AUC(all var): {auc}')

l = y_train.shape[0]
oof_odds = np.ones(l) * 1 / 9
for j in range(var_len):
    if roc_auc_score(y_train, oof[:, j]) >= 0.500:
        oof_odds *= (9 * oof[:, j] / (1 - oof[:, j]))

auc = roc_auc_score(y_train, oof_odds)
print(f'AUC(th0.5): {auc}')

sub_train = pd.DataFrame(zip(y_train, oof_odds), columns=['y','p'])
sub_train.sort_values('p', ascending=False, inplace=True)

for i in range(100, 2000, 100):
    sub_train_ = sub_train.head(i)
    print(i, accuracy_score(sub_train_['y'], sub_train_['p']>0))


# save raw pred
np.save(f'../data/{__file__}_oof', oof)
np.save(f'../data/{__file__}_p_test_all', p_test_all)

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

print(sub.target.describe())

os.system(f'gsutil cp {SUBMIT_FILE_PATH} gs://malware_onodera/')
os.system(f'cp LOG/log_{__file__}.txt LOG/log_{__file__}_{SEED}.txt')
os.system(f'gsutil cp LOG/log_{__file__}_{SEED}.txt gs://malware_onodera/')


"""
gsutil cp gs://malware_onodera/*.gz ../output/
gsutil cp gs://malware_onodera/*.txt LOG/
"""


#==============================================================================
utils.end(__file__)
utils.stop_instance()

