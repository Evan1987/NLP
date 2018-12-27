
"""
coding: utf-8
@date: 2018-12-27
@author: Cigar
Tensorflow implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling"
((https://arxiv.org/abs/1609.01454))
https://github.com/applenob/RNN-for-Joint-NLU
"""

import numpy as np
import numpy.ma as ma  # masked array
from sklearn.metrics import f1_score, accuracy_score

def _acc(true_data, pred_data, true_length=None):
    """
    计算acc值
    :param true_data: 真实值 [#sample, #seq]
    :param pred_data: 预测值 [#sample, #seq]
    :param true_length: 有效长度 [#sample]
    :return: scalar float
    """
    true_data = np.asarray(true_data)
    pred_data = np.asarray(pred_data)
    assert true_data.shape == pred_data.shape, "Shape not match between true and pred!"
    if true_length is not None:
        total_num = np.sum(true_length)
        assert total_num > 0, "All true length is 0, please check!"
        res = 0
        for _true, _pred, _length in zip(true_data, pred_data, true_length):
            res += np.sum(_true[: _length] == _pred[: _length], axis=None)
    else:
        total_num = np.prod(true_data.shape)  # 行数 * 列数，总元素量
        assert total_num > 0, "There is no true value, please check!"
        res = np.sum(true_data == pred_data, axis=None)  # sum all elements
    
    return res / float(total_num)


def get_data_from_sequence_batch(true_batch, pred_batch, padding_token):
    """
    从序列batch中提取数据
    batch: [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]
    """
    true_ma = ma.masked_equal(true_batch, padding_token)
    mask = true_ma.mask
    pred_ma = ma.masked_array(pred_batch, mask)
    true_ma = true_ma[~mask]
    pred_ma = pred_ma[~mask]
    return true_ma.data, pred_ma.data


def f1_for_sequence_batch(true_batch, pred_batch, average="micro", padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    labels = list(set(true))
    return f1_score(true, pred, labels=labels, average=average)

def acc_for_sequence_batch(true_batch, pred_batch, padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    return _acc(true, pred)

