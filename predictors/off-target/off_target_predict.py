# -*-coding: utf-8 -*-
"""
@author: jianfeng yan
@license: python3.8
@contact: yanjianfeng@westlakegenetech.edu.cn
@software: PyCharm
@file: off_target_predict.py
@time:
@desc:
"""
from off_target_features import *
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


##################################################################
# System parameters
rnn_params = {'bilstm_hidden1': 32,
              'bilstm_hidden': 64,
              'hidden1': 64,
              'dropout': 0.2276}

on_model_directory_dict = {'K562': '../models/on-target/k562/on-target_RNN-weights_for-K562',
                           'Jurkat': '../models/on-target/jurkat/on-target_RNN-weights_for-Jurkat'}

off_model_path_dict = {'K562': '../models/off-target/off-target-best-model-for-K562.model'}

off_target_params_dict = {'K562': ('all', '+P+M')}
##################################################################


# step 1: to predict target sequence on-target scores
# obtain sequence data for dataframe
def obtain_Sequence_data(data, layer_label='1D'):
    """
    input: dataframe with 'target sequence' column
    (63bp: 20bp downstream + 20bp target + 3bp pam + 20bp upstream)
    """
    x_data = []
    for i, row in data.iterrows():
        try:
            seq = row['target sequence']
            # assert seq[41:43] == "GG"
            one_sample = obtain_each_seq_data(seq)
        except AttributeError as e:
            raise e
        if layer_label == '1D':  # for LSTM or Conv1D, shape=(sample, step, feature)
            one_sample_T = one_sample.T
            x_data.append(one_sample_T)
        else:
            x_data.append(one_sample)
    x_data = np.array(x_data)
    if layer_label == '2D':  # for Conv2D shape=(sample, rows, cols, channels)
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
    else:
        pass # for LSTM or Conv1D: shape=(sample, step, feature)
    x_data = x_data.astype('float32')
    # print('After transformation, x_data.shape:', x_data.shape)
    return x_data


# RNN
# input 63bp length sequence
def RNN(params, seq_len=63):
    from keras.models import Model
    from keras.layers import LSTM, Bidirectional
    from keras.layers import Input
    from keras.layers import Dense, Dropout
    # Model Frame
    visible = Input(shape=(seq_len, 4))
    bi_lstm1 = Bidirectional(LSTM(params['bilstm_hidden1'], dropout=0.2, return_sequences=True))(visible)
    bi_lstm = Bidirectional(LSTM(params['bilstm_hidden'], dropout=0.2))(bi_lstm1)
    hidden1 = Dense(params['hidden1'], activation='relu')(bi_lstm)
    dropout = Dropout(params['dropout'])(hidden1)
    output = Dense(1)(dropout)
    # model architecture
    model = Model(inputs=visible, outputs=output)
    return model


# get on-target features
def get_on_target_features(cell, data):
    wtdata = data[['target sequence']]
    otdata = data[['off-target sequence']]
    otdata.rename(columns={'off-target sequence': 'target sequence'}, inplace=True)
    wtdata.drop_duplicates(inplace=True)
    otdata.drop_duplicates(inplace=True)
    wtdata.reset_index(drop=True, inplace=True)
    otdata.reset_index(drop=True, inplace=True)
    # on-target score
    model_directory = on_model_directory_dict[cell]
    model = RNN(rnn_params, seq_len=63)
    model.load_weights(model_directory).expect_partial()
    x_wt = obtain_Sequence_data(wtdata, layer_label='1D')
    x_ot = obtain_Sequence_data(otdata, layer_label='1D')
    ypred_wt = model.predict(x_wt)
    ypred_ot = model.predict(x_ot)
    wtdata['on_pred'] = ypred_wt
    otdata['off_pred'] = ypred_ot
    otdata.rename(columns={'target sequence': 'off-target sequence'}, inplace=True)
    # merge
    data = pd.merge(data, wtdata, how='left', on='target sequence')
    data = pd.merge(data, otdata, how='left', on='off-target sequence')
    data.reset_index(drop=True, inplace=True)
    return data


# get features of off-target models
# input columns: the 'target sequence' column represents 63bp wild-type sequence;
# the 'off-target sequence' column represents 63bp off-target sequence.
def off_target_predict(cell, data):
    fixed_feat, feat_label = off_target_params_dict[cell]
    # features
    data = get_on_target_features(cell, data)
    x_data = off_target_mismatch_feature_engineering(data, fixed_feat, feat_label)
    x_data = np.array(x_data)
    # predict
    import joblib
    off_model_path = off_model_path_dict[cell]
    model = joblib.load(off_model_path)
    ypred = model.predict(x_data)
    return ypred


# main
def main(input_path, output_dir):
    cell = 'K562'
    mkdir(output_dir)
    output_path = output_dir + '/predicted_result_Aidit_OFF_%s.txt' % cell
    is_Exist_file(output_path)
    with open(output_path, 'a') as a:
        wline = 'target sequence\toff-target sequence\toff-target_score\n'
        a.write(wline)
        with open(input_path) as f:
            next(f)
            batch_n = 100000
            i = 0
            batch_data_dict = {'target sequence': [],
                               'off-target sequence': []}
            for line in f:
                i += 1
                if i <= batch_n:
                    line = line.strip(' ').strip('\n')
                    wtseq, otseq = line.split('\t')
                    batch_data_dict['target sequence'].append(wtseq)
                    batch_data_dict['off-target sequence'].append(otseq)
                else:
                    # predict
                    batch_data = pd.DataFrame(batch_data_dict)
                    batch_ypred = off_target_predict(cell, batch_data)
                    batch_data['off-target_score'] = batch_ypred
                    for index, row in batch_data.iterrows():
                        wline = '%s\t%s\t%s\n' % (row['target sequence'], row['off-target sequence'], row['off-target_score'])
                        a.write(wline)
                    # initial
                    i = 0
                    batch_data_dict = {'target sequence': [],
                                       'off-target sequence': []}
            # last predict
            batch_data = pd.DataFrame(batch_data_dict)
            batch_ypred = off_target_predict(cell, batch_data)
            batch_data['off-target_score'] = batch_ypred
            for index, row in batch_data.iterrows():
                wline = '%s\t%s\t%s\n' % (row['target sequence'], row['off-target sequence'], row['off-target_score'])
                a.write(wline)


if __name__ == '__main__':
    # parameter: cell, input_path, output_dir
    # input columns:
    # 'target sequence': 63bp wild-type sequence (20bp upstream + 20bp target + 3bp PAM + 20bp downstream);
    # 'off-target sequence': 63bp off-target sequence (20bp upstream + 20bp off-target + 3bp PAM + 20bp downstream).
    input_path, output_dir = sys.argv[1:]
    # input_path = "./demo_dataset.txt"
    # output_dir = "./result"
    main(input_path, output_dir)

