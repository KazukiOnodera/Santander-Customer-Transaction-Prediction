#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:35:13 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
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
    
    utils.end(__file__)


