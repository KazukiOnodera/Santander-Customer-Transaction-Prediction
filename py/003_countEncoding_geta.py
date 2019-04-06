#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:22:00 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils

PREF = 'f003'


dirs  = [f'../data/var_{i:03}' for i in range(200)]
var_names = [f'var_{i:03}' for i in range(200)]

d_v = list(zip(dirs, var_names))

def output(df, name):
    """
    name: 'train' or 'test'
    """
    
    for d,v in tqdm(d_v):
        df.filter(regex=f'^(?=.*{v}).*$').to_pickle(f'{d}/{name}_{PREF}.pkl')
    
    return


def fe(df):
    
    feature = pd.DataFrame(index=df.index)
    
    for GETA in range(1,10):
        for i in [3,2,1,0]:
            p = (10 ** i)
            geta = GETA / (10 ** (i+1))
            for c in tqdm(df.columns):
                s = ((df[c] + geta) * p * 2 + 1) // 2 / p # round
                di = s.value_counts().to_dict()
                feature[f'{PREF}_{c}_g{GETA}_r{i}'] = s.map(di)
    
    tr_ = feature.iloc[:200000]
    output(tr_, 'train')
    
    te_ = feature.iloc[200000:].reset_index(drop=True)
    output(te_, 'test')
    
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


