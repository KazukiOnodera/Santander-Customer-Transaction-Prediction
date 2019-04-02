#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:16:51 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
import utils

PREF = 'f006'

est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')

def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    df = pd.DataFrame(est.fit_transform(df), columns=df.columns)
    
    for c in tqdm(df.columns):
        di = df[c].value_counts().to_dict()
        feature[f'{PREF}_{c}'] = df[c].map(di)
    
#    for i in [3,2,1]:
#        for c in tqdm(df.columns):
#            di = df[c].round(i).value_counts().to_dict()
#            feature[f'{PREF}_{c}_r{i}'] = df[c].round(i).map(di)
    
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
    te.drop(np.load('../data/fake_index.npy'), inplace=True)
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


