#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:01:04 2019

@author: kazuki.onodera
"""

from time import time
from datetime import datetime
import os
from socket import gethostname
HOSTNAME = gethostname()

# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    print(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return

def reset_time():
    global st_time
    st_time = time()
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    print(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return

def elapsed_minute():
    return (time() - st_time)/60

