#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:00:01 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f012'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    li = []
    df_ = df.T.copy()
    for c in tqdm(df_.columns):
        di = df_[c].value_counts().to_dict()
        li.append( (df_[c].map(di)==1).sum() )
    
    feature[f'{PREF}_sum'] = li
    
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


