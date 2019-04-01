#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:33:54 2019

@author: kazuki.onodera

is only target==1

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f012'

def only_target1(c):
    li = list( set(tr1[c]. unique()) - set(tr0[c].unique()) )
    return li

def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    for c in tqdm(df.columns):
        li = only_target1(c)
        feature[f'{PREF}_{c}'] = (df[c].isin(li))*1
    
    feature[f'{PREF}_sum'] = feature.sum(1)
    
    feature.iloc[:200000].to_pickle(f'../data/train_{PREF}.pkl')
    feature.iloc[200000:].reset_index(drop=True).to_pickle(f'../data/test_{PREF}.pkl')
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    y_train = utils.load_target()['target']
    te = utils.load_test().drop(['ID_code'], axis=1)
    
    tr0 = tr[y_train==0]
    tr1 = tr[y_train==1]
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


