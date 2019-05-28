#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:15:38 2019

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


# =============================================================================
# our team progress
# =============================================================================
df = pd.read_csv('/Users/kazuki.onodera/Downloads/santander-customer-transaction-prediction-publicleaderboard.csv',
                 parse_dates=['SubmissionDate'])



df = df[df['TeamName']=='三人寄れば文殊の知恵（本当か？']
df = df[df['SubmissionDate'].between(pd.to_datetime('2019-03-11'), pd.to_datetime('2019-04-11'))]


df.sort_values('SubmissionDate', inplace=True)

df.set_index('SubmissionDate', inplace=True)


ix = df.index.tolist()
ax = df.Score.plot()
ax.yaxis.tick_right()
plt.xticks(ix[::3]+[ix[-1]], rotation='vertical', fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('SubmissionDate', fontsize=13)
plt.title('Public LB Score Progress', fontsize=15)

plt.hlines(0.92425, min(ix), max(ix), "gold", linestyles='dashed')
plt.hlines(0.90120, min(ix), max(ix), "silver", linestyles='dashed')
plt.hlines(0.90116, min(ix), max(ix), "brown", linestyles='dashed')
plt.tight_layout()

