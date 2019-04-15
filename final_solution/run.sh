#!/bin/sh

# =============================================================================
# link from rawdata(includes .zip and .csv)
# =============================================================================
mkdir ../onodera/input
mkdir ../akiyama/input

ln __RAWDATA__/* onodera/input/
ln __RAWDATA__/* akiyama/input/

# =============================================================================
# onodera part
# =============================================================================
cd onodera/py
mkdir ../data
mkdir ../output

python -u 000_init.py  > LOG/log_000_init.py.txt



