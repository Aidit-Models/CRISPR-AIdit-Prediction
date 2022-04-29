# -*-coding: utf-8 -*-
"""
@author: jianfeng yan
@license: python3.8
@contact: yanjianfeng@westlakegenetech.edu.cn
@software: PyCharm
@file: DSB_repair_predict.py
@time:
@desc: DSB_repair Repair Prediction
"""
import sys
from data import *

##############################################################################
# model parameters
DSB_model_params_dict = {
     'K562': ['../models/DSB_repair/Integrated_K562_DSB_Repair_Merged_Information.csv',
              '../models/DSB_repair/K562/XGB_K562-63bp_%s.model',
              '../models/DSB_repair/K562/DSB_repair-outcomes-ensamble-best-weight-for-K562.hdf5'],

     'Jurkat': ['../models/DSB_repair/Integrated_Jurkat_DSB_Repair_Merged_Information.csv',
                '../models/DSB_repair/Jurkat/XGB_Jurkat-63bp_%s.model',
                '../models/DSB_repair/Jurkat/DSB_repair-outcomes-ensamble-best-weight-for-Jurkat.hdf5']}
##############################################################################

# Get train & test data
##############################################################################
def Obtain_predicting_feature_2nd(data, seq_bp=28, max_len=30):
    # 1. to get sequence feature
    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)
    # 2. to get MH feature
    edit_sites = [34, 35, 36, 37, 38, 39, 40]
    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)
    return (seq_data, MH_data)


# For XGBoost Ensamble
def xgb_prediction(Xdata, model_path):
    import joblib
    model = joblib.load(model_path)  # 加载
    ypred = model.predict(Xdata)
    return ypred


def main_xgb_prediction(int_data, data, model_path_pattern, seq_bp=63, max_len=30):
    # Get Xdata
    seq_data, MH_data = Obtain_predicting_feature_2nd(data, seq_bp, max_len)
    Xdata = pd.merge(seq_data, MH_data, how='inner', on=['target sequence'])
    del Xdata['target sequence']
    Xdata = np.array(Xdata)
    print('----------------------')
    print('Xtrain.shape:', Xdata.shape)

    # Get Engineered Feature
    # model_path = "XGB_K562-63bp_%s.model"%("29:40D-12")
    eng_data = seq_data[['target sequence']]
    for model_label in int_data['new category'].unique():
        model_label = model_label.replace(':', '_')
        temp_model_path = model_path_pattern%(model_label)
        ypred = xgb_prediction(Xdata, temp_model_path)
        eng_data[model_label] = ypred
    return (seq_data, MH_data, eng_data)
##############################################################################


# prediction
##############################################################################
# 自定义损失函数
def my_categorical_crossentropy_2(labels, logits):
    import tensorflow as tf
    '''
    label = tf.constant([[0,0,1,0,0]], dtype=tf.float32)
    logit = tf.constant([[-1.2, 2.3, 4.1, 3.0, 1.4]], dtype=tf.float32)
    logits = tf.nn.softmax(logit) # 计算softmax
    my_result1 = my_categorical_cross_entropy(labels=label, logits=logits)
    my_result2 = my_categorical_crossentropy_1(label, logits)
    my_result3 = my_categorical_crossentropy_2(label, logits)
    my_result1, my_result2, my_result3
    '''
    return tf.keras.losses.categorical_crossentropy(labels, logits)


def prediction(model_path, Xdata):
    # load model
    from keras.models import load_model
    model = load_model(model_path, custom_objects={'my_categorical_crossentropy_2': my_categorical_crossentropy_2})
    # prediction & evluation
    ypred = model.predict(Xdata)
    return ypred


def batch_predict(batch_data, int_data, xgb_model_path, ensamble_model_path):
    seq_data, MH_data, eng_data = main_xgb_prediction(int_data, batch_data, xgb_model_path, seq_bp=63)
    feat_list = [seq_data, MH_data, eng_data]
    Xdata = pd.concat([temp_data.iloc[:, 1:] for temp_data in feat_list], axis=1)
    Xdata = np.array(Xdata)
    # prediction
    ypred = prediction(ensamble_model_path, Xdata)
    ypred = pd.DataFrame(ypred)
    ypred.columns = list(int_data['new category'].unique())
    ypred = pd.concat([seq_data[['target sequence']], ypred], axis=1)
    return ypred


def write_content(data, data_path):
    cols = data.columns.tolist()
    if os.path.exists(data_path):
        line_format = '%s' + '\t%s' * (data.shape[0] - 1) + '\n'
        with open(data_path, 'a') as a:
            for index, row in data.iterrows():
                line = line_format % tuple([row[col] for col in cols])
                a.write(line)
    else:
        data.to_csv(data_path, sep='\t', index=False)


# for ensemble prediction
def main(cell, input_path, output_dir):
    # parameters
    int_data_path, xgb_model_path, ensamble_model_path = DSB_model_params_dict[cell]
    int_data = pd.read_csv(int_data_path)
    # read
    mkdir(output_dir)
    output_path = output_dir + '/predicted_result_Aidit_DSB_%s.txt' % cell
    is_Exist_file(output_path)
    with open(input_path, 'r') as f:
        next(f)
        batch_n = 100000
        i = 0
        batch_data_dict = {'target sequence': []}
        for line in f:
            i += 1
            if i <= batch_n:
                line = line.strip(' ').strip('\n')
                wtseq = line.split('\t')[0]
                batch_data_dict['target sequence'].append(wtseq)
            else:
                # predict
                batch_data = pd.DataFrame(batch_data_dict)
                ypred = batch_predict(batch_data, int_data, xgb_model_path, ensamble_model_path)
                write_content(ypred, output_path)
                # initial
                i = 0
                batch_data_dict = {'target sequence': []}
        # predict
        batch_data = pd.DataFrame(batch_data_dict)
        ypred = batch_predict(batch_data, int_data, xgb_model_path, ensamble_model_path)
        write_content(ypred, output_path)
##############################################################################


# Prediction -- K562
###############################################################################
if __name__ == '__main__':
    # input columns: 'target sequence': 63bp (20bp upstream + 20bp target + 3bp PAM + 20bp downstream)
    cell, input_path, output_dir = sys.argv[1:]
    # cell = "K562"
    # input_path = "./demo_dataset.txt"
    # output_dir = "./result"
    main(cell, input_path, output_dir)

