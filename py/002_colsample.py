#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:45:36 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
#from itertools import combinations
import utils

PREF = 'f002'

LOOP = 99
SIZE = 10
SEED = 1
np.random.seed(SEED)



def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    for i in tqdm(range(LOOP)):
        col = np.random.choice(df.columns, size=SIZE)
        feature[f'{PREF}_{i}_min'] = df[col].min(1)
        feature[f'{PREF}_{i}_mean'] = df[col].mean(1)
        feature[f'{PREF}_{i}_max'] = df[col].max(1)
        feature[f'{PREF}_{i}_std'] = df[col].std(1)
    
    feature.iloc[:200000].to_pickle(f'../data/train_{PREF}.pkl')
    feature.iloc[200000:].to_pickle(f'../data/test_{PREF}.pkl')
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    te = utils.load_test().drop(['ID_code'], axis=1)
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


