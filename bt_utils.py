#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : bt_utils.py
@Time    : 2021/08/12 15:30:05
@Author  : HickeyHsy
@Contact : hickeyhsu@163.com
@Version : 0.1
@License : Apache License Version 2.0, January 2004
@Desc    : 回测工具的一些计算函数
'''

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'==='] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

def mad_filter(x,high,low):
    if x>high:
        return high
    elif x<low:
        return low
    else:
        return x 
def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
    return data


def MaxDrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list)- return_list)/np.maximum.accumulate(return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])
    return(return_list[j] - return_list[i]) / return_list[j],j,i

def standard(metrics2mad,bond_metric_DF:pd.DataFrame):
    k=3 #拦截倍数   
    for metric in metrics2mad:
        # 计算拦截值
        
        mad=bond_metric_DF[metric].mad()#计算mad值
        median=bond_metric_DF[metric].median()# 计算中位值
        high=median+k*mad#高值
        low=median-k*mad#低值
        var=bond_metric_DF[metric].var()# 计算方差
        #MAD拦截
        cname="{}_MAD".format(metric)
        bond_metric_DF[cname]=bond_metric_DF[metric].apply(func=mad_filter,args=(high,low))
        # print("{}:{},{},{},{},{}".format(metric,mad,median,high,low,var))

        #zscore计算
        zname="{}_zcores".format(metric)
        zscaler = preprocessing.StandardScaler()
        scale_param=zscaler.fit(bond_metric_DF[cname].values.reshape(-1, 1))
        bond_metric_DF[zname] = zscaler.fit_transform(bond_metric_DF[cname].values.reshape(-1, 1), scale_param)

        #min-max归一化
        sname="{}_std".format(metric)
        minmax_scaler= preprocessing.MinMaxScaler()
        minmax_scale_param=minmax_scaler.fit(bond_metric_DF[zname].values.reshape(-1, 1))
        bond_metric_DF[sname] = minmax_scaler.fit_transform(bond_metric_DF[zname].values.reshape(-1, 1), minmax_scale_param)
        # bond_metric_DF.to_csv('V8.csv')
    return bond_metric_DF