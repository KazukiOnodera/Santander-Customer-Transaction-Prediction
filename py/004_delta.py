#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:44:08 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import utils

PREF = 'f004'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    comb = list(combinations(df.columns, 2))
    for c1,c2 in tqdm(comb):
        feature[f'{PREF}_{c1}-m-{c2}'] = (df[c1] - df[c2]).astype(np.float32)
    
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


