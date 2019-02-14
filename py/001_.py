#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:45:36 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from itertools import combinations
import utils


def fe(df):
    
    comb = list(combinations(df.columns, 2))
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    tr = utils.load_train().drop(['ID_code', 'target'], axis=1)
    fe(tr)
    
    te = utils.load_test().drop(['ID_code'], axis=1)
    fe(tr)
    
    
    
    utils.end(__file__)


