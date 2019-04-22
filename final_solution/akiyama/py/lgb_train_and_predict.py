#!/usr/bin/env python
# coding: utf-8
import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

class ModelExtractionCallback(object):
    """
    original author : momijiame
    ref : https://blog.amedama.jp/entry/lightgbm-cv-model
    description : Class for callback to extract trained models from lightgbm.cv(). 
    note: This class depends on private class '_CVBooster', so there are some future risks. 
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # Booster へのプロキシオブジェクトを返す
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # Booster のリストを返す
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # Early stop したときの boosting round を返す
        return self._model.best_iteration
    
def arrange_dataset(df, cnum):
    _dset = df.filter(regex=f'var_{cnum}$')
    _dset.columns = list(range(_dset.shape[1]))
    _dset = _dset.assign(var_num = cnum)
    return _dset
    
def main():
    model_output_dir = f'../processed/lgb_output/'
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)

    dataset_dir = '../processed/dataset/'
    X_train = pd.read_pickle(os.path.join(dataset_dir, 'X_train.pickle'))
    y_train = pd.read_pickle(os.path.join(dataset_dir, 'y_train.pickle'))
    X_test = pd.read_pickle(os.path.join(dataset_dir, 'X_test.pickle'))

    params = {
        'bagging_freq': 5,
        'bagging_fraction': 0.95,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 1.0,
        'learning_rate': 0.005,
        'max_depth': -1,
        'metric':'binary_logloss',
        'min_data_in_leaf': 30,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 64,
        'num_threads': 32,
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': 1}

    dset_list = []
    for cnum in range(200):
        _dset = arrange_dataset(X_train, cnum)
        dset_list.append(_dset)

    concat_X_train = pd.concat(dset_list, axis=0)
    concat_X_train['var_num'] = concat_X_train['var_num'].astype('category')

    train_dset = lgb.Dataset(
        concat_X_train, 
        pd.concat([y_train for c in range(200)], axis=0), 
        free_raw_data=False
    )

    for fold_set_number in range(10):
        print('### start iter {} in 10 ###'.format(fold_set_number+1))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019+fold_set_number)
        folds = [
            [
                np.concatenate([_trn+i * X_train.shape[0] for i in range(200)]), 
                np.concatenate([_val+i * X_train.shape[0] for i in range(200)])
            ] for _trn, _val in skf.split(X_train, y_train)]

        extraction_cb = ModelExtractionCallback()
        callbacks = [extraction_cb,]

        print('start training. ')
        cv_result = lgb.cv(params, train_set=train_dset, num_boost_round=100000, 
                                 early_stopping_rounds=100, verbose_eval=100, folds=folds, callbacks=callbacks)
        bsts = extraction_cb.raw_boosters
        best_iteration = extraction_cb.best_iteration
        print('training end. ')

        print('start predicting. ')
        oof_pred_array = np.ones((X_train.shape[0], 200))
        test_pred_array = np.ones((X_test.shape[0], 5, 200))
        for cnum in tqdm(range(200)):
            for i, bst in enumerate(bsts):
                cv_valid_index = bst.valid_sets[0].used_indices
                cv_valid_index = cv_valid_index[:int(cv_valid_index.shape[0]/200)]
                # oofの予測
                cv_valid_data = arrange_dataset(X_train, cnum).iloc[cv_valid_index].values
                oof_pred_array[cv_valid_index, cnum] = bst.predict(cv_valid_data, num_iteration=best_iteration)
                # testの予測
                test_pred_array[:, i, cnum] = bst.predict(arrange_dataset(X_test, cnum).values, num_iteration=best_iteration)
        print('prediction end. ')

        print('start postprocess. ')
        thr = 0.500
        oof_pred_odds_prod = np.ones((X_train.shape[0]))
        test_pred_odds_prod = np.ones((X_test.shape[0], 5))
        for cnum in tqdm(range(200)):
            tmp_auc = roc_auc_score(y_train, oof_pred_array[:, cnum])
            if tmp_auc >= thr:
                oof_pred_odds_prod *= oof_pred_array[:, cnum] / (1 - oof_pred_array[:, cnum])
                test_pred_odds_prod *= test_pred_array[:,:, cnum] / (1 - test_pred_array[:,:, cnum])
        print('postprocess end. auc : {0:.6f}'.format(roc_auc_score(y_train, oof_pred_odds_prod)))

        print('save iteration results')
        pd.DataFrame(oof_pred_odds_prod, index=X_train.index, columns=['pred'])\
            .to_pickle(os.path.join(model_output_dir, f'oof_preds_{fold_set_number}.pkl.gz'), compression='gzip')
        for fold_num in range(5):
            model_management_num = fold_num + fold_set_number*5
            pd.DataFrame(test_pred_odds_prod[:, fold_num], index=X_test.index, columns=['pred'])\
                .to_pickle(os.path.join(model_output_dir, f'test_preds_{model_management_num}.pkl.gz'), compression='gzip')

if __name__ == '__main__':
    main()
