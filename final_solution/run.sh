#!/bin/sh

# =============================================================================
# link from rawdata(csv.zip)
# =============================================================================
mkdir onodera/input
mkdir akiyama/input

ln __RAWDATA__/* onodera/input/
ln __RAWDATA__/* akiyama/input/

# =============================================================================
# onodera part
# =============================================================================
cd onodera/py
mkdir ../data
mkdir ../output
mkdir LOG

python -u 000_init.py > LOG/log_000_init.py.txt

python -u 906_predict_0410-1.py 7441  > LOG/log_906_predict_0410-1.py_7441.txt
python -u 906_predict_0410-1.py 11834 > LOG/log_906_predict_0410-1.py_11834.txt
python -u 906_predict_0410-1.py 17670 > LOG/log_906_predict_0410-1.py_17670.txt
python -u 906_predict_0410-1.py 27017 > LOG/log_906_predict_0410-1.py_27017.txt
python -u 906_predict_0410-1.py 42687 > LOG/log_906_predict_0410-1.py_42687.txt
python -u 906_predict_0410-1.py 59352 > LOG/log_906_predict_0410-1.py_59352.txt
python -u 906_predict_0410-1.py 63018 > LOG/log_906_predict_0410-1.py_63018.txt
python -u 906_predict_0410-1.py 73770 > LOG/log_906_predict_0410-1.py_73770.txt
python -u 906_predict_0410-1.py 84474 > LOG/log_906_predict_0410-1.py_84474.txt
python -u 906_predict_0410-1.py 95928 > LOG/log_906_predict_0410-1.py_95928.txt

python -u 907_predict_0410-2.py 37170  > LOG/log_907_predict_0410-2.py_37170.txt
python -u 907_predict_0410-2.py 68389  > LOG/log_907_predict_0410-2.py_68389.txt
python -u 907_predict_0410-2.py 71946  > LOG/log_907_predict_0410-2.py_71946.txt
python -u 907_predict_0410-2.py 75783  > LOG/log_907_predict_0410-2.py_75783.txt
python -u 907_predict_0410-2.py 81857  > LOG/log_907_predict_0410-2.py_81857.txt
python -u 907_predict_0410-2.py 85689  > LOG/log_907_predict_0410-2.py_85689.txt
python -u 907_predict_0410-2.py 93117  > LOG/log_907_predict_0410-2.py_93117.txt


# =============================================================================
# akiyama part
# =============================================================================
cd ../../akiyama/py

python -u make_features.py
python -u lgb_train_and_predict.py
python -u lgb_postprocess.py
python -u nn_train_and_predict.py
python -u nn_postprocess.py


# =============================================================================
# blending part
# =============================================================================
cd ../../onodera/py

python -u 911_blend_0410-6.py  > LOG/log_911_blend_0410-6.py.txt

