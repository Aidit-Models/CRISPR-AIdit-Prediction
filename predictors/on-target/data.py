# -*-coding: utf-8 -*-
'''
@author: jianfeng yan
@license: python3.8
@contact: yanjianfeng@westlakegenetech.edu.cn
@software: PyCharm
@file: data.py
@time:
@desc: feature for on-target
input sequence: 63bp (20bp upstream + 20bp target + 3bp PAM + 20bp downstream)
'''
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def is_Exist_file(path):
    import os
    if os.path.exists(path):
        os.remove(path)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Create directory successfully.')
    else:
        print(path + ' the directory exists.')


# input: target directory
# output: complete path (path + file name)
def walk(path):
    input_path_list = []
    if not os.path.exists(path):
        return -1
    for root, dirs, names in os.walk(path):
        for filename in names:
            input_path = os.path.join(root, filename)
            input_path_list.append(input_path)
    return input_path_list


# input： sequence features
###########################################
def find_all(sub, s):
    index = s.find(sub)
    feat_one = np.zeros(len(s))
    while index != -1:
        feat_one[index] = 1
        index = s.find(sub, index + 1)
    return feat_one


# obtain single sequence
def obtain_each_seq_data(seq):
    A_array = find_all('A', seq)
    G_array = find_all('G', seq)
    C_array = find_all('C', seq)
    T_array = find_all('T', seq)
    one_sample = np.array([A_array, G_array, C_array, T_array])
    # print(one_sample.shape)
    return one_sample


# obtain sequence data for dataframe
def obtain_Sequence_data(data, layer_label='1D'):
    '''
    input: dataframe with 'target sequence' column
    (63bp: 20bp downstream + 20bp target + 3bp pam + 20bp upstream)
    '''
    x_data = []
    for i, row in data.iterrows():
        try:
            seq = row['target sequence']
            assert seq[41:43] == "GG"
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


# for single sequence
def obtain_single_sequence_data(seq):
    x_data = []
    assert (seq[41:43] == 'GG') & (len(seq) == 63)
    one_sample = obtain_each_seq_data(seq)
    one_sample_T = one_sample.T
    x_data.append(one_sample_T)
    x_data = np.array(x_data)
    x_data = x_data.astype('float32')
    return x_data
