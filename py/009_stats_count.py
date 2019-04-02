#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:00:12 2019

@author: kazuki.onodera
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f009'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    col = df.columns
    feature[f'{PREF}_min'] = df[col].min(1)
    feature[f'{PREF}_mean'] = df[col].mean(1)
    feature[f'{PREF}_max'] = df[col].max(1)
    feature[f'{PREF}_std'] = df[col].std(1)
    feature[f'{PREF}_median'] = df[col].median(1)
    feature[f'{PREF}_max-min'] = feature[f'{PREF}_max'] - feature[f'{PREF}_min']
    
    col = feature.columns
    
    for c in tqdm(feature.columns):
        di = feature[c].value_counts().to_dict()
        feature[f'{c}_cnt'] = feature[c].map(di)
    
    feature.drop(col, axis=1, inplace=True)
    
    feature.iloc[:200000].to_pickle(f'../data/train_{PREF}.pkl')
    feature.iloc[200000:].reset_index(drop=True).to_pickle(f'../data/test_{PREF}.pkl')
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    te = utils.load_test().drop(['ID_code'], axis=1)
    te = te.drop(np.load('../data/fake_index.npy'))
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


