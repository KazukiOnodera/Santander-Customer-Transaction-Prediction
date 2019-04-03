#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:15:46 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import utils

PREF = 'f005'

sc = MinMaxScaler()

def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    df_ = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
    
#    for c in tqdm(df.columns):
#        di = df[c].value_counts().to_dict()
#        feature[f'{PREF}_{c}'] = df[c].map(di)
    
    for i in [3,2,1,0]:
        for c in tqdm(df.columns):
            di = df_[c].round(i).value_counts().to_dict()
            feature[f'{PREF}_{c}_r{i}'] = df_[c].round(i).map(di)
    
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


