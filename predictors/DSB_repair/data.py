# -*-coding: utf-8 -*-
"""
@author: jianfeng yan
@license: python3.8
@contact: yanjianfeng@westlakegenetech.edu.cn
@software: PyCharm
@file: data.py
@time:
@desc: DSB_repair features
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# 基础功能 1：删除文件和创建文件夹
#########################################################################
# 检查文件是否存在，存在删除
def is_Exist_file(path):
    import os
    if os.path.exists(path):
        os.remove(path)


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
##########################################################################


# 获取单位置核苷酸特征
# 基础功能 2：  Get sequence Feature
#########################################################################
# get_dummies: feature one-hot encoding
def Get_Dummies(df, feature_list):
    df_dummies = pd.get_dummies(df[feature_list], columns=feature_list, prefix_sep='-')
    ## 去除 df 中含有 df_dummies 的列
    for col in df_dummies.columns.tolist():
        if col in df.columns.tolist():
            del df[col]
    ## concat
    df = pd.concat([df, df_dummies], axis=1)
    return df


def helper_single_feature_list(raw_data, seq_bp):
    raw_data['gRNAUp'] = raw_data['target sequence'].apply(lambda x: x[:20])
    raw_data['gRNATarget'] = raw_data['target sequence'].apply(lambda x: x[20:40])
    raw_data['PAM'] = raw_data['target sequence'].apply(lambda x: x[40:43])
    raw_data['gRNADown'] = raw_data['target sequence'].apply(lambda x: x[43:63])
    # 单位置核苷酸特征
    single_feature_list = []
    if seq_bp == 63:
        # Up
        for i in range(20):
            raw_data['S-U%s' % (i + 1)] = raw_data['gRNAUp'].apply(lambda x: x[i])
            single_feature_list.append('S-U%s' % (i + 1))
        # Target
        for i in range(20):
            raw_data['S-T%s' % (i + 1)] = raw_data['gRNATarget'].apply(lambda x: x[i])
            single_feature_list.append('S-T%s' % (i + 1))
        # PAM
        raw_data['S-PAM(N)'] = raw_data['PAM'].apply(lambda x: x[0])
        single_feature_list.append('S-PAM(N)')
        # Down
        for i in range(20):
            raw_data['S-D(-%s)' % (i + 1)] = raw_data['gRNADown'].apply(lambda x: x[i])
            single_feature_list.append('S-D(-%s)' % (i + 1))
    else:  # 28bp
        # Target
        for i in range(20):
            raw_data['S-T%s' % (i + 1)] = raw_data['gRNATarget'].apply(lambda x: x[i])
            single_feature_list.append('S-T%s' % (i + 1))
        # PAM
        raw_data['S-PAM(N)'] = raw_data['PAM'].apply(lambda x: x[0])
        single_feature_list.append('S-PAM(N)')
        # Down
        for i in range(5):
            raw_data['S-D(-%s)' % (i + 1)] = raw_data['gRNADown'].apply(lambda x: x[i])
            single_feature_list.append('S-D(-%s)' % (i + 1))
    del raw_data['gRNAUp']
    del raw_data['gRNATarget']
    del raw_data['PAM']
    del raw_data['gRNADown']
    return single_feature_list


def obtain_single_sequence_one_hot_feature_2nd(data, seq_bp):
    import time
    import copy

    raw_data = copy.deepcopy(data)

    print('================================')
    print('Function: Obtain_Single_Sequence_One_Hot_Feature ... ...')
    s = time.time()
    raw_data['target sequence'] = raw_data['target sequence'].apply(lambda x: x + "GTTTGTATTACCGCCATGCATT")
    single_feature_list = helper_single_feature_list(raw_data, seq_bp)
    raw_data = Get_Dummies(raw_data, single_feature_list)
    # check all one-hot features in raw_data.columns & complement
    import copy
    single_one_hot_feature_list = []
    nfeat_list = copy.deepcopy(raw_data.columns.tolist())
    for feat in single_feature_list:
        for nucle in ['A', 'C', 'G', 'T']:
            one_hot_feat = feat + '-' + nucle
            single_one_hot_feature_list.append(one_hot_feat)
            if one_hot_feat not in nfeat_list:  # 补充不完整的 one-hot 特征
                raw_data[one_hot_feat] = 0
        del raw_data[feat]  # 删除非 one-hot 过渡特征
    raw_data = raw_data[['target sequence'] + single_one_hot_feature_list]
    e = time.time()
    print("Using Time: %s" % (e - s))
    print('================================\n')
    return raw_data
##########################################################################


# 获取微同源特征
# 基础功能 3：  Get MH Feature
#############################################################################
# Deletion Classes
def deletion_classes(edit_sites, max_len=30):
    min_site = min(edit_sites)
    max_site = max(edit_sites)
    delt_classes = []
    for i in range(1, max_len):
        inf_site = min_site - i + 1
        for site in range(inf_site, max_site + 1):
            key = '%s:%sD-%s' % (site, site + i - 1, i)
            delt_classes.append(key)
    delt_classes.append('D%s+' % (max_len))
    return delt_classes


# get MH feature
# Get 1bp MH
def help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq):
    delt_nucle = gRNASeq[delt_inf_site]
    MH_nucle = gRNASeq[delt_sup_site + 1]
    if delt_nucle == MH_nucle:
        MH = 1
    else:
        MH = 0
    return MH


# Get 2bp MH
def help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq):
    delt_nucles = gRNASeq[delt_inf_site:(delt_inf_site + 2)]
    MH_nucles = gRNASeq[(delt_sup_site + 1):(delt_sup_site + 3)]
    if delt_nucles == MH_nucles:
        MH = 1
    else:
        MH = 0
    return MH


# Get 3bp MH
def help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq):
    delt_nucles = gRNASeq[delt_inf_site:(delt_inf_site + 3)]
    MH_nucles = gRNASeq[(delt_sup_site + 1):(delt_sup_site + 4)]
    if delt_nucles == MH_nucles:
        MH = 1
    else:
        MH = 0
    return MH


# Get 4bp MH
def help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq):
    delt_nucles = gRNASeq[delt_inf_site:(delt_inf_site + 4)]
    MH_nucles = gRNASeq[(delt_sup_site + 1):(delt_sup_site + 5)]
    if delt_nucles == MH_nucles:
        MH = 1
    else:
        MH = 0
    return MH


# MH: 1bp, 2bp, 3bp
# get MH feature
def Get_MH_Feature(gRNASeq, delt_classes, max_len=30):
    MH_feat_dict = {}
    for one_class in delt_classes:
        if one_class != 'D%s+' % (max_len):
            delt_len = int(one_class.split('-')[1])
            delt_p = one_class.split('-')[0]
            delt_inf_site = int(delt_p.split(':')[0]) - 1
            delt_sup_site = int(delt_p.split(':')[1][:-1]) - 1
            if delt_len == 1:
                MH = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH_feat_dict['%s_1bp' % (one_class)] = MH
            elif delt_len == 2:
                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH_feat_dict['%s_1bp' % (one_class)] = MH1
                MH_feat_dict['%s_2bp' % (one_class)] = MH2
            elif delt_len == 3:
                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH_feat_dict['%s_1bp' % (one_class)] = MH1
                MH_feat_dict['%s_2bp' % (one_class)] = MH2
                MH_feat_dict['%s_3bp' % (one_class)] = MH3
            else:
                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH4 = help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq)
                MH_feat_dict['%s_1bp' % (one_class)] = MH1
                MH_feat_dict['%s_2bp' % (one_class)] = MH2
                MH_feat_dict['%s_3bp' % (one_class)] = MH3
                MH_feat_dict['%s_4bp' % (one_class)] = MH4
        else:
            pass
    # sorting
    keys = list(MH_feat_dict.keys())
    keys.sort(reverse=False)
    MH_feats = [MH_feat_dict[key] for key in keys]
    return (MH_feats, keys)


# adjust two gRNASeq_85bp
def adjust_column_gRNASeq_85bp(gRNA_name, gRNASeq_85bp):
    if (gRNA_name != 'AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG') & (
            gRNA_name != 'AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC'):
        return gRNASeq_85bp
    else:
        if (gRNA_name == 'AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG') & (gRNASeq_85bp is np.nan):
            return 'AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT'
        elif (gRNA_name == 'AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC') & (gRNASeq_85bp is np.nan):
            return 'AGCTATAGGTCCAAGGGCTCAGCGCTGATGGGCTCAGCGCTGATGGGCTCAGCGCTGGGCTTGAGAGCAGGAGTGTGTGTTTGTATTACCGCCATGCATT'
        else:
            return gRNASeq_85bp


def assertion(seq):
    assert seq[41:43] == 'GG'
    return seq


# 主函数: Get MH feature
def main_MH_Feature_2nd(data, edit_sites, max_len=30):
    import copy
    df = copy.deepcopy(data)
    # adjust two gRNASeq_85bp
    df['target sequence'] = df['target sequence'].apply(lambda x: x + "GTTTGTATTACCGCCATGCATT")
    df['target sequence'] = df['target sequence'].apply(lambda x: assertion(x))
    # get mh features
    delt_classes = deletion_classes(edit_sites, max_len)
    df['MH_features'] = df['target sequence'].apply(lambda x: Get_MH_Feature(x, delt_classes, max_len)[0])
    MH_data = pd.DataFrame(list(np.array(df['MH_features'])))
    # get columns
    gRNASeq_85bp = 'AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT'
    cols = Get_MH_Feature(gRNASeq_85bp, delt_classes, max_len)[1]
    MH_data.columns = cols
    del df['MH_features']
    df = pd.concat([df[['target sequence']], MH_data], axis=1)
    return df
##############################################################################


