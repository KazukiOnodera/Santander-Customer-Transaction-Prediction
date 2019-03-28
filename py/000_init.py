#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:35:13 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import utils



# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = pd.read_csv('../input/train.csv.zip')
    te = pd.read_csv('../input/test.csv.zip')
    
    tr.to_pickle('../data/train.pkl')
    tr[['target']].to_pickle('../data/target.pkl')
    te.to_pickle('../data/test.pkl')
    
    
    # MinMaxScaler
    trte = pd.concat([tr, te], ignore_index=True)[tr.columns]
    
    sc = MinMaxScaler()
    
    trte_ = pd.DataFrame(sc.fit_transform(trte.iloc[:, 2:]), 
                         columns=tr.columns[2:])
    trte_ = pd.concat([trte[['ID_code', 'target']], trte_], axis=1)
    
    tr_ = trte_[trte_.target.notnull()]
    te_ = trte_[trte_.target.isnull()]
    del te_['target']
    tr_.to_pickle('../data/train_min0max1.pkl')
    te_.to_pickle('../data/test_min0max1.pkl')
    
    
    
    
    utils.end(__file__)


