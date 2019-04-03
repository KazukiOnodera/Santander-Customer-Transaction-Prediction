#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:17:44 2019

@author: kazuki.onodera

count stats

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f008'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    for c in tqdm(df.columns):
        di = df[c].value_counts().to_dict()
        feature[f'{PREF}_{c}'] = df[c].map(di)
    
    col = feature.columns
    feature[f'{PREF}_min'] = feature[col].min(1)
    feature[f'{PREF}_mean'] = feature[col].mean(1)
    feature[f'{PREF}_max'] = feature[col].max(1)
    feature[f'{PREF}_std'] = feature[col].std(1)
    feature[f'{PREF}_median'] = feature[col].median(1)
    feature[f'{PREF}_max-min'] = feature[f'{PREF}_max'] - feature[f'{PREF}_min']
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


