#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:49:08 2019

@author: Kazuki

is peak

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f005'


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    for c in tqdm(df.columns):
        di = tr1[c].round(1).value_counts().to_dict()
        feature[f'{PREF}_{c}'] = df[c].round(1).map(di)
    
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
    
    y_train = utils.load_target()['target']
    tr1 = tr[y_train==1]
    
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    fe(trte)
    
    
    utils.end(__file__)


