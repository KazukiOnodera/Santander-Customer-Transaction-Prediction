#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:53:19 2019

@author: kazuki.onodera

OHE

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import cpu_count, Pool
import utils

PREF = 'f003'


dirs  = [f'../data/var_{i:03}' for i in range(200)]
var_names = [f'var_{i:03}' for i in range(200)]

d_v = list(zip(dirs, var_names))


def multi(args):
    d, v = args
    feature = pd.get_dummies(trte[v].round(0)).add_prefix(f'{PREF}_{v}_')
    
    feature.iloc[:200000].to_pickle(f'{d}/train_{PREF}.pkl')
    feature.iloc[200000:].reset_index(drop=True).to_pickle(f'{d}/test_{PREF}.pkl')
    
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
    del tr, te; gc.collect()
    
    pool = Pool(cpu_count())
    pool.map(multi, d_v)
    pool.close()
    
    utils.end(__file__)


