#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
import numpy as np
'''
1，生成一个一个的batch
2，将每个sentence转化为id
3，加上unk，start，end
4，pad
5，返回长度
'''

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
source_path = father_path + '/data/cut1000.zh'
target_path = father_path + '/data/small1000.en'
source_vocab = father_path + '/data/source_vocab.txt'
source_vec = father_path + '/data/source_vec.txt'
target_vocab = father_path + '/data/target_vocab.txt'
target_vec = father_path + '/data/target_vec.txt'
unk = 'uuunnnkkk'
tgt_sos_id = '\<s>'
tgt_eos_id = '\</s>'
batch_size = 32
max_length = 200

s_voc = []
s_vec = []
t_voc = []
t_vec = []


def vocab_parse(path, vocab):
    for line in open(path):
        vocab.append(line.strip())


def vec_parse(path, vec):
    for line in open(path):
        vector = line.strip()
        vec.append(list(map(float, vector.split())))


vocab_parse(source_vocab, s_voc)
vocab_parse(target_vocab, t_voc)
vec_parse(source_vec, s_vec)
vec_parse(target_vec, t_vec)


def find_index(vocab, word):
    if word in vocab:
        return vocab.index(word)
    else:
        return vocab.index(unk)


def sentence_token(sentence, which_for):
    words = sentence.strip().split(' ')
    newsentence = []
    if (which_for == 'source_input'):
        vocab = s_voc
        for word in words:
            newsentence.append(find_index(vocab, word))
    elif (which_for == 'target_input'):
        vocab = t_voc
        start_id = t_voc.index(tgt_sos_id)
        newsentence.append(start_id)
        for word in words:
            newsentence.append(find_index(vocab, word))
    elif (which_for == 'target_output'):
        vocab = t_voc
        end_id = t_voc.index(tgt_eos_id)
        newsentence.append(end_id)
        for word in words:
            newsentence.append(find_index(vocab, word))
    return newsentence


def pad_sentence_func(sentence, max_length):
    sentence_length = len(sentence)
    pad_length = max_length - sentence_length
    pad_sentence = sentence + [0] * pad_length
    return (pad_sentence, sentence_length)


def generate_data():
    encoder_inputs = []
    encoder_length = []
    decoder_inputs = []
    decoder_length = []
    decoder_outputs = []
    f1 = open(source_path)
    for line in f1:
        sentence = sentence_token(line, 'source_input')
        pad_sentence, sentence_length = pad_sentence_func(sentence, max_length)
        encoder_inputs.append(pad_sentence)
        encoder_length.append(sentence_length)
    f2 = open(target_path)
    for line in f2:
        sentence_inputs = sentence_token(line, 'target_input')
        pad_sentence_inputs, sentence_length = pad_sentence_func(
            sentence_inputs, max_length)
        decoder_inputs.append(pad_sentence_inputs)
        decoder_length.append(sentence_length)
        sentence_outputs = sentence_token(line, 'target_output')
        pad_sentence_outputs, sentence_length = pad_sentence_func(
            sentence_outputs, max_length)
        decoder_outputs.append(pad_sentence_outputs)
    f1.close()
    f2.close()
    return(encoder_inputs, decoder_inputs, decoder_outputs, encoder_length, decoder_length)


def shuffle_aligned_list(data1, data2, data3, data4, data5):
    num = len(data1)
    p = np.random.permutation(num)
    return ([data1[i] for i in p], [data2[i] for i in p], [data3[i] for i in p], [data4[i] for i in p], [data5[i] for i in p])


def batch_generator(data1, data2, data3, data4, data5, batch_size, shuffle=True):
    if shuffle:
        data1, data2, data3, data4, data5 = shuffle_aligned_list(
            data1, data2, data3, data4, data5)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data1):
            batch_count = 0
            if shuffle:
                data1, data2, data3, data4, data5 = shuffle_aligned_list(
                    data1, data2, data3, data4, data5)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield (data1[start:end], data2[start:end], data3[start:end], data4[start:end], data5[start:end])


def generate_batch(batch_size):
    encoder_inputs, decoder_inputs, decoder_outputs, encoder_length, decoder_length = generate_data()
    gen = batch_generator(encoder_inputs, decoder_inputs,
                          decoder_outputs, encoder_length, decoder_length, batch_size)
    return gen


if __name__ == '__main__':
    gen = generate_batch(batch_size)
    encoder_inputs_batch, decoder_inputs_batch, decoder_outputs_batch, encoder_length_batch, decoder_length_batch = next(
        gen)
