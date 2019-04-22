#!/usr/bin/env python
# coding: utf-8
import os

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Dense, BatchNormalization,Dropout, Embedding, Flatten, Concatenate, Input
from keras.layers.core import Lambda
from keras import backend as K
from keras import optimizers
from keras.layers.advanced_activations import PReLU

from scipy.special import erfinv
class RankGaussScalar(object):
    """
    usage: 
    rgs = RankGaussScalar()
    rgs.fit(df_X)
    df_X_converted = rgs.transform(df_X)
    df_X_test_converted = rgs.transform(df_X_test)
    """
    def __init__(self):
        self.fit_done = False

    def rank_gauss(self, x):
        N = x.shape[0]
        temp = x.argsort()
        rank_x = temp.argsort() / N
        rank_x -= rank_x.mean()
        rank_x *= 2
        efi_x = erfinv(rank_x)
        efi_x -= efi_x.mean()
        return efi_x

    def fit(self, df_x):
        """
        df_x: fitting対象のDataFrame
        """
        self.train_unique_rankgauss = {}
        self.target_cols = np.sort(df_x.columns)
        for c in self.target_cols:
            unique_val = np.sort(df_x[c].unique())
            self.train_unique_rankgauss[c]= [unique_val, self.rank_gauss(unique_val)]
        self.fit_done = True

    def transform(self, df_target):
        """
        df_target: transform対象のDataFrame
        """
        assert self.fit_done
        assert np.all(np.sort(np.intersect1d(df_target.columns, self.target_cols)) == np.sort(self.target_cols))
        df_converted_rank_gauss = pd.DataFrame(index=df_target.index)
        for c in self.target_cols:
            df_converted_rank_gauss[c] = np.interp(df_target[c], 
                                                   self.train_unique_rankgauss[c][0], 
                                                   self.train_unique_rankgauss[c][1]) # ,left=0, right=0)
        return df_converted_rank_gauss

def build_model():
    
    model_input = Input(shape=(6,))
    num_input = Lambda(lambda x: x[:, :5], output_shape=(5,))(model_input)
    cat_input = Lambda(lambda x: x[:, 5:], output_shape=(1,))(model_input)
    
    x = num_input
    x = Dense(256)(num_input)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)

    emb_dim = 128
    y = Embedding(200, emb_dim, input_length=1)(cat_input)
    y = Flatten()(y)
    
    z = Concatenate()([x, y])
    z = Dense(256)(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)
    z = Dropout(rate=0.5)(z)
    z = Dense(256)(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)
    z = Dropout(rate=0.5)(z)
    z = Dense(256)(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)
    output = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=model_input, outputs=output)
    return model

def arrange_dataset(df, cnum):
    _dset = df.filter(regex=f'var_{cnum}$')
    _dset.columns = list(range(_dset.shape[1]))
    _dset = _dset.assign(var_num = cnum)
    return _dset

def main():
    model_output_dir = f'../processed/nn_output/'
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)

    dataset_dir = '../processed/dataset/'
    X_train = pd.read_pickle(os.path.join(dataset_dir, 'X_train.pickle'))
    y_train = pd.read_pickle(os.path.join(dataset_dir, 'y_train.pickle'))
    X_test = pd.read_pickle(os.path.join(dataset_dir, 'X_test.pickle'))

    epochs = 40
    batch_size = 1024
    patience = 5

    dset_list = []
    for cnum in range(200):
        _dset = arrange_dataset(X_train, cnum)
        dset_list.append(_dset)
    concat_X_train = pd.concat(dset_list, axis=0)
    train_dset = [concat_X_train, pd.concat([y_train for c in range(200)], axis=0)]

    weights_dir = '../processed/keras_weights'
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    for fold_set_number in range(10):
        print('### start iter {} in 10 ###'.format(fold_set_number+1))
        K.clear_session()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019+fold_set_number)
        folds = [
            [
                np.concatenate([_trn+i * X_train.shape[0] for i in range(200)]), 
                np.concatenate([_val+i * X_train.shape[0] for i in range(200)])
            ] for _trn, _val in skf.split(X_train, y_train)]

        for fold_num in range(5):
            print(f'## Start KFold number {fold_num} ##')
            model_management_num = fold_num + fold_set_number*5
            skf_train_index, skf_valid_index = folds[fold_num]

            nonprogress_counter=0
            e_auc_best = 0

            weight_path = os.path.join(weights_dir, f'{model_management_num}.model')
            skf_X_train = train_dset[0].iloc[skf_train_index].copy()
            skf_y_train = train_dset[1].iloc[skf_train_index]
            skf_X_valid = train_dset[0].iloc[skf_valid_index].copy()
            skf_y_valid = train_dset[1].iloc[skf_valid_index]
            single_valid_index = skf_valid_index[:skf_valid_index.shape[0]//200]  

            rgscaler = RankGaussScalar()
            rgscaler.fit(skf_X_train.iloc[:, :5].astype(float))
            skf_X_train.iloc[:, :5] = rgscaler.transform(skf_X_train.iloc[:, :5].astype(float))
            skf_X_valid.iloc[:, :5] = rgscaler.transform(skf_X_valid.iloc[:, :5].astype(float))

            print('start training. ')
            for _e in range(epochs):
                print('epoch {}'.format(_e))
                if _e == 0:
                    model = build_model()
                    #optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-4)
                    optimizer = optimizers.adam(lr=0.001)
                    model.compile(loss='binary_crossentropy', optimizer=optimizer)

                history = model.fit(skf_X_train.values, skf_y_train,
                                validation_data=[skf_X_valid.values, skf_y_valid], 
                                epochs=1,
                                batch_size=batch_size,
                                shuffle=True,
                                verbose=1)

                oof_pred_array = np.ones((single_valid_index.shape[0], 200))
                for cnum in range(200):
                    oof_pred_array[:, cnum] = np.squeeze(
                        model.predict(
                            skf_X_valid.iloc[cnum*single_valid_index.shape[0]:(cnum+1)*single_valid_index.shape[0]].values, batch_size=100000
                        )
                    )
                e_auc = roc_auc_score(y_train.iloc[single_valid_index], oof_pred_array.prod(axis=1))

                print('\tauc : {0:.6f}'.format(e_auc))
                if e_auc > e_auc_best:
                    model.save_weights(weight_path)
                    e_auc_best = e_auc
                    nonprogress_counter = 0
                else:
                    nonprogress_counter += 1

                if (nonprogress_counter >= patience) or (_e == (epochs-1)):
                    print('fold end. ')
                    break
            print('training end. ')

            model.load_weights(weight_path)

            print('start predicting. ')
            oof_pred_array = np.ones((single_valid_index.shape[0], 200))
            test_pred_array = np.ones((X_test.shape[0], 200))
            for cnum in range(200):
                tmp_X_test = arrange_dataset(X_test, cnum).copy()
                tmp_X_test.iloc[:, :5] = rgscaler.transform(tmp_X_test.iloc[:, :5].astype(float))
                oof_pred_array[:, cnum] = np.squeeze(
                    model.predict(
                        skf_X_valid.iloc[cnum*single_valid_index.shape[0]:(cnum+1)*single_valid_index.shape[0]].values, batch_size=100000
                    )
                )
                test_pred_array[:, cnum] = np.squeeze(model.predict(tmp_X_test.values, batch_size=100000))
            fold_oof_pred = pd.DataFrame(oof_pred_array, index=X_train.index[single_valid_index])
            fold_test_pred = pd.DataFrame(test_pred_array, index=X_test.index)
            print('prediction end. ')

            print('save fold results')
            fold_oof_pred.to_pickle(os.path.join(model_output_dir, f'oof_preds_{model_management_num}.pkl.gz'), compression='gzip')
            fold_test_pred.to_pickle(os.path.join(model_output_dir, f'test_preds_{model_management_num}.pkl.gz'), compression='gzip')

if __name__ == '__main__':
    main()
