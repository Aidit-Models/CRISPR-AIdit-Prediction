# -*-coding: utf-8 -*-
'''
@author: jianfeng yan
@license: python3.8
@contact: yanjianfeng@westlakegenetech.edu.cn
@software: PyCharm
@file: on_target_predict.py
@time:
@desc: for on-target
input sequence: 63bp (20bp upstream + 20bp target + 3bp PAM + 20bp downstream)
'''
import sys
from data import *
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

##############################################################################
# System parameters

rnn_params = {'bilstm_hidden1': 32,
              'bilstm_hidden': 64,
              'hidden1': 64,
              'dropout': 0.2276}

model_directory_dict = {
                        'K562': '../models/on-target/k562/on-target_RNN-weights_for-K562',
                        'Jurkat': '../models/on-target/jurkat/on-target_RNN-weights_for-Jurkat'
                       }
##############################################################################


# custom evaluation function
def get_spearman_rankcor(y_true, y_pred):
    from scipy.stats import spearmanr
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))


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


# for batch sequence
# data with 'target sequence' column
def predict_batch_data(model_directory, input_path, output_path):
    # load model
    model = RNN(rnn_params, seq_len=63)
    model.load_weights(model_directory).expect_partial()
    is_Exist_file(output_path)
    with open(output_path, 'a') as a:
        with open(input_path, 'r') as f:
            a.write("target sequence\tefficiency\n")
            next(f)
            for line in f:
                line = line.strip(' ').strip('\n')
                seq = line.split('\t')[0]
                try:
                    x_data = obtain_single_sequence_data(seq)
                except AssertionError as e:
                    raise e
                eff = model.predict(x_data)
                a.write("%s\t%s\n" % (seq, np.array(eff)[0][0]))


# for batch sequence
# data with 'target sequence' column
def predict_batch_data_2(model_directory, input_path, output_path):
    # load model
    model = RNN(rnn_params, seq_len=63)
    model.load_weights(model_directory).expect_partial()
    is_Exist_file(output_path)
    with open(output_path, 'a') as a:
        with open(input_path, 'r') as f:
            a.write("target sequence\tefficiency\n")
            next(f)
            batch_n = 100000
            i = 0
            x_data = []
            seq_list = []
            for line in f:
                i += 1
                line = line.strip(' ').strip('\n')
                seq = line.split('\t')[0]
                if i <= batch_n:
                    assert (seq[41:43] == 'GG') & (len(seq) == 63)
                    one_sample = obtain_each_seq_data(seq)
                    one_sample_T = one_sample.T
                    seq_list.append(seq)
                    x_data.append(one_sample_T)
                else:
                    # predict
                    x_data = np.array(x_data)
                    x_data = x_data.astype('float32')
                    ypred = model.predict(x_data)
                    for index, seq in enumerate(seq_list):
                        eff = ypred[index][0]
                        a.write("%s\t%s\n" % (seq, eff))
                    # Re-initial
                    i, x_data, seq_list = 0, [], []
                    assert (seq[41:43] == 'GG') & (len(seq) == 63)
                    one_sample = obtain_each_seq_data(seq)
                    one_sample_T = one_sample.T
                    seq_list.append(seq)
                    x_data.append(one_sample_T)
            # predict
            x_data = np.array(x_data)
            x_data = x_data.astype('float32')
            ypred = model.predict(x_data)
            for index, seq in enumerate(seq_list):
                eff = ypred[index][0]
                a.write("%s\t%s\n" % (seq, eff))


# predict
def main_predict(args):
    if len(args[1:]) == 2: # for single sequence input
        cell_line, sequence = args[1:]
        model_directory = model_directory_dict[cell_line]
        model = RNN(rnn_params, seq_len=63)
        model.load_weights(model_directory).expect_partial()
        seq = sequence.strip(' ')
        x_data = obtain_single_sequence_data(seq)
        eff = model(x_data)
        print("sequence input: %s" % sequence)
        print("efficiency generated by Aidit_ON_%s: %s" % (cell_line, np.array(eff)[0][0]))
        return eff
    elif len(args[1:]) == 3:  # for batch prediction
        cell_line, input_path, output_directory = args[1:]
        model_directory = model_directory_dict[cell_line]
        mkdir(output_directory)
        output_path = output_directory + '/predicted_result_Aidit_ON_%s.txt' % cell_line
        # predict_batch_data(model_directory, input_path, output_path)
        predict_batch_data_2(model_directory, input_path, output_path)
    else:
        print("Please check arguments, not match with demo instruction.")


# prediction
if __name__ == "__main__":
    main_predict(sys.argv)



