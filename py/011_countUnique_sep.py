#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:33:46 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f011'


def fe(df, name):
    
    feature = pd.DataFrame(index=df.index)
    
    for c in tqdm(df.columns):
        di = df[c].value_counts().to_dict()
        feature[f'{PREF}_{c}'] = (df[c].map(di)==1)*1
    
    feature[f'{PREF}_sum'] = feature.sum(1)
    
    feature.to_pickle(f'../data/{name}_{PREF}.pkl')
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    te = utils.load_test().drop(['ID_code'], axis=1)
    
    fe(tr, 'train')
    fe(te, 'test')
    
    
    utils.end(__file__)


