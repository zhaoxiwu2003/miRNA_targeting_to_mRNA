# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:58:48 2020

@author: Xiwu Zhao
"""
import re
import random
from abc import ABC
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


def loadTxtData(filename, skip_line_numbers=0):
    seqs = list()
    list()
    len_miRNA: int = 0
    len_mRNA = 0
    with open(filename, 'r') as infl:
        infl = infl.readlines()[skip_line_numbers:]
        for line in infl:
            value = line.split('\t')
            # values = re.sub('T','U',value[1])
            values = [elem.strip('\r\n') for elem in value]
            encode = dict(zip('NAUCG', range(5)))
            name = values[0] + '_' + values[2]
            values[1] = re.sub('T', 'U', value[1])
            values[3] = re.sub('T', 'U', value[3])
            seq1 = [encode[s] for s in values[1]]
            len_miRNA = len(seq1)
            seq2 = [encode[s] for s in values[3]]
            len_mRNA = len(seq2)
            label = int(values[4])
            seq = (seq1, seq2, len_miRNA, len_mRNA, label, name)
            seqs.append(seq)
    return seqs


def padding_sequence(seqs):
    """
    This function is used to padding
    00...000+miRNA+000.000+mRNA+000.0000
    """
    seqs_padded = list()
    padding_numbers = list()
    labels = list()
    max_len = 0
    # len_conca = [seq[2]+seq[3] for seq in seqs]
    # max_len = max(len_conca)
    max_len = 100
    print('Max integrated length of miRNA and mRNA', max_len)
    for seq in seqs:
        padding_1 = list()
        padding_2 = list()
        padding_3 = list()
        padding_total_len = max_len - len(seq[0]) - len(seq[1])
        if padding_total_len == 0:
            padding_1 = [0]
            padding_2 = [0]
            padding_3 = [0]
        else:
            list_1 = list(range(0, padding_total_len))
            padding_1 = random.sample(list_1, 1)
            if padding_1 == padding_total_len:
                padding_2 = [0]
                padding_3 = [0]
            else:
                list_2 = list(range(0, padding_total_len - padding_1[0]))
                padding_2 = random.sample(list_2, 1)
                if padding_2 == padding_total_len - padding_1[0]:
                    padding_3 = [0]
                else:
                    padding_3 = [padding_total_len - padding_1[0] - padding_2[0]]
        padding_number = [padding_1, padding_2, padding_3]
        padding_numbers.append(padding_number)
        labels.append(seq[4])
        seqs_temp = [0] * padding_1[0] + seq[0] + [0] * padding_2[0] + seq[1] + [0] * padding_3[0]
        seqs_padded.append(seqs_temp)
    return seqs_padded, labels, max_len, padding_numbers


class MyModel(Model, ABC):
    def __init__(self):
        super(MyModel, self).__init__()
        self.my_model = Sequential([
            layers.Embedding(input_dim=5, output_dim=5, input_length=100),
            layers.Conv1D(filters=320, kernel_size=12, activation='relu'),
            layers.Dropout(0.2),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            layers.LSTM(32, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2)])

    @tf.function
    def call(self, inputs, training=None):
        out = self.my_model(inputs)
        return out
