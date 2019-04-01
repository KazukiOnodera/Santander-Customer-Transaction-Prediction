#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:43:49 2019

@author: Kazuki

is peak

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f006'

col = ['var_12', 'var_108', 'var_126', 'var_181']

def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    for c in tqdm(col):
        v = df[c].round(3).value_counts().head(1).index[0]
        feature[f'{PREF}_{c}'] = (df[c].round(3)==v)*1
    
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
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


