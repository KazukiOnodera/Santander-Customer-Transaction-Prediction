#!/usr/bin/env python
# coding: utf-8
import os

import pandas as pd
import numpy as np

from tqdm import tqdm

def find_real_id(test):
    '''
    original author : yag320 
    ref : https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
    '''
    test_array = test.drop(['ID_code'], axis=1).values

    unique_count = np.zeros_like(test_array)
    for feature in tqdm(range(test_array.shape[1])):
        _, index_, count_ = np.unique(test_array[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
    
    real_id = test.loc[real_samples_indexes, 'ID_code'].values
    fake_id = test.loc[synthetic_samples_indexes, 'ID_code'].values
    
    return real_id, fake_id

def reverse_and_scale(df):
    '''
    original author : jiweiliu
    ref : https://www.kaggle.com/jiweiliu/fast-pdf-calculation-with-correlation-matrix
    '''
    
    def reverse(df):
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
        reverse_list = ['var_%d'%i for i in reverse_list]
        for col in tqdm(reverse_list):
            df[col] = df[col]*(-1)
        return df

    def scale(df):
        for col in tqdm(df.columns):
            if col.startswith('var_'):
                mean,std = df[col].mean(),df[col].std()
                df[col] = (df[col]-mean)/std
        return df
    
    df = reverse(df)
    df = scale(df)
    return df

def round_and_count(all_df, fake_id):
    def my_round(val, digit=0):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p
    
    real_df = all_df.drop(fake_id, axis=0)
    
    feature_list = []
    
    _feature = all_df.copy()
    for c in tqdm(real_df.columns):
        count_dict = real_df[c].value_counts().to_dict()
        _feature[c] = all_df[c].map(count_dict)
    _feature = _feature.add_prefix('concat_count_')
    feature_list.append(_feature)
    
    for digit in range(4, 1, -1):
        _feature = all_df.copy()
        for c in tqdm(all_df.columns):
            all_rounded = my_round(all_df[c], digit)
            real_rounded = my_round(real_df[c], digit)
            count_dict = real_rounded.value_counts().to_dict()
            _feature[c] = all_rounded.map(count_dict)
        _feature = _feature.add_prefix(f'concat_count_round{digit}_')
        feature_list.append(_feature)
    
    concat_feature = pd.concat(feature_list, axis=1)
    
    return concat_feature

def main():
    input_dir = '../input/'
    train = pd.read_csv(os.path.join(input_dir, 'train.csv.zip'))
    test = pd.read_csv(os.path.join(input_dir, 'test.csv.zip'))
    concat_df = pd.concat([train.drop('target', axis=1), test], axis=0, sort=False).set_index('ID_code')
    train_id = train['ID_code']
    test_id = test['ID_code']

    real_id, fake_id = find_real_id(test)
    scaled_df = reverse_and_scale(concat_df)
    count_df = round_and_count(scaled_df, fake_id)

    all_feature_df = pd.concat([scaled_df, count_df], axis=1)

    dataset_dir = '../processed/dataset/'
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    all_feature_df.loc[train_id].to_pickle(os.path.join(dataset_dir, 'X_train.pickle'))
    all_feature_df.loc[test_id].to_pickle(os.path.join(dataset_dir, 'X_test.pickle'))
    train.set_index('ID_code').loc[train_id, 'target'].to_pickle(os.path.join(dataset_dir, 'y_train.pickle'))
    pd.to_pickle(real_id, os.path.join(dataset_dir, 'real_id.pickle'))
    pd.to_pickle(fake_id, os.path.join(dataset_dir, 'fake_id.pickle'))

if __name__ == '__main__':
    main()
