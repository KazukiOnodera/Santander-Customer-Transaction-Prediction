#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:23:29 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import utils

PREF = 'f004'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    df_ = df.round(0)
    
    comb = list(combinations(df.columns, 2))
    for c1,c2 in tqdm(comb, mininterval=30):
        cnt = df_.groupby([c1, c2]).size().to_frame()
        feature[f'{PREF}_{c1}-&-{c2}'] = pd.merge(df_[[c1, c2]], cnt, how='left', on=[c1, c2])[0]
    
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


