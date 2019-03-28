#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:43:03 2019

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

PREF = 'f001'


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    tr.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    
    te = utils.load_test().drop(['ID_code'], axis=1)
    te.add_prefix(PREF+'_').to_pickle(f'../data/test_{PREF}.pkl')
    
    
    utils.end(__file__)

