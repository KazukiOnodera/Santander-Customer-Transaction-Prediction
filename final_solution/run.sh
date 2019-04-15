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
mkdir ../LOG

python -u 000_init.py  > LOG/log_000_init.py.txt

python -u 906_predict_0410-1.py 7441  > LOG/log_906_predict_0410-1_7441.py.txt
python -u 906_predict_0410-1.py 11834 > LOG/log_906_predict_0410-1_11834.py.txt
python -u 906_predict_0410-1.py 17670 > LOG/log_906_predict_0410-1_17670.py.txt
python -u 906_predict_0410-1.py 27017 > LOG/log_906_predict_0410-1_27017.py.txt
python -u 906_predict_0410-1.py 42687 > LOG/log_906_predict_0410-1_42687.py.txt
python -u 906_predict_0410-1.py 59352 > LOG/log_906_predict_0410-1_59352.py.txt
python -u 906_predict_0410-1.py 63018 > LOG/log_906_predict_0410-1_63018.py.txt
python -u 906_predict_0410-1.py 73770 > LOG/log_906_predict_0410-1_73770.py.txt
python -u 906_predict_0410-1.py 84474 > LOG/log_906_predict_0410-1_84474.py.txt
python -u 906_predict_0410-1.py 95928 > LOG/log_906_predict_0410-1_95928.py.txt



