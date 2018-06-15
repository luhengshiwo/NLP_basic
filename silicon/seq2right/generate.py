#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import time
import os
import numpy as np
from parameters import Para
import jieba
'''
1，生成一个一个的batch
2，将每个sentence转化为id
3，加上unk，start，end
4，pad
5，返回长度
'''
source_vocab = 'data/vocab.txt'
source_vec = 'data/vec.txt'
target_vocab = 'data/vocab.txt'
target_vec = 'data/vec.txt'
unk = Para.unk
pad_id = Para.pad
# pad_id = 0
batch_size = Para.batch_size
tgt_sos = Para.tgt_sos
tgt_eos = Para.tgt_eos

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
        start_id = t_voc.index(tgt_sos)
        newsentence.append(start_id)
        for word in words:
            newsentence.append(find_index(vocab, word))
    elif (which_for == 'target_output'):
        vocab = t_voc
        end_id = t_voc.index(tgt_eos)
        for word in words:
            newsentence.append(find_index(vocab, word))
        newsentence.append(end_id)
    return newsentence


def pad_sentence_batch(sentences):  # pad_int是<pad>的索引
    pad_int = find_index(s_voc, pad_id)
    max_sentence = max([len(sentence) for sentence in sentences])
    newsentences = []
    sentence_length = []
    for sentence in sentences:
        newsentences.append(sentence + [pad_int]
                            * (max_sentence - len(sentence)))
        sentence_length.append(len(sentence))
    return (newsentences, sentence_length)


def shuffle_aligned_list(data0, data1, data2, data3):
    num = len(data1)
    p = np.random.permutation(num)
    return ([data0[i] for i in p], [data1[i] for i in p], [data2[i] for i in p], [data3[i] for i in p])


def generate_batch(data_path, batch_size, isdev=False, shuffle=True):
    data_file = open(data_path)
    # target_file = open(target_path)
    ids = []
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    for line in data_file:
        lines = line.strip().split('\t')
        ids.append(lines[0])
        sentence = sentence_token(lines[1], 'source_input')
        encoder_inputs.append(sentence)
        sentence_input = sentence_token(lines[2], 'target_input')
        sentence_output = sentence_token(lines[2], 'target_output')
        decoder_inputs.append(sentence_input)
        decoder_outputs.append(sentence_output)
    data_file.close()
    # target_file.close()
    if shuffle:
        ids, encoder_inputs, decoder_inputs, decoder_outputs = shuffle_aligned_list(
            ids, encoder_inputs, decoder_inputs, decoder_outputs)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(encoder_inputs):
            batch_count = 0
            if shuffle:
                ids, encoder_inputs, decoder_inputs, decoder_outputs = shuffle_aligned_list(
                    ids, encoder_inputs, decoder_inputs, decoder_outputs)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        encoder_batch_pad, encoder_seq_length = pad_sentence_batch(
            encoder_inputs[start:end])
        decoder_batch_pad, decoder_seq_length = pad_sentence_batch(
            decoder_inputs[start:end])
        target_batch_pad, _ = pad_sentence_batch(
            decoder_outputs[start:end])
        yield (encoder_batch_pad, decoder_batch_pad, target_batch_pad, encoder_seq_length, decoder_seq_length)

def generate_batch_shell(sentence, batch_size, isdev=False, shuffle=True):
    encoder_inputs = []   
    for _ in range(batch_size):
        new_sentence = " ".join(jieba.cut(sentence))
        encoder_inputs.append(sentence_token(new_sentence, 'source_input'))
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(encoder_inputs):
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        encoder_batch_pad, encoder_seq_length = pad_sentence_batch(
            encoder_inputs[start:end])
        yield (encoder_batch_pad, encoder_seq_length)






def embeddings():
    return(s_vec, t_vec, s_voc, t_voc)


def id2words(batch_sentence_ids):
    sentences = []
    # print(s_voc)
    for sentences_ids in batch_sentence_ids:
        words = []
        for word_id in sentences_ids:
            word = t_voc[word_id]
            if word == pad_id:
                break
            words.append(word)
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences


def id2words_source(batch_sentence_ids):
    sentences = []
    for sentences_ids in batch_sentence_ids:
        words = []
        for word_id in sentences_ids:
            word = t_voc[word_id]
            if word == pad_id:
                break
            words.append(word)
    sentences.append([words])
    return sentences


def id2words_target(batch_sentence_ids):
    sentences = []
    for sentences_ids in batch_sentence_ids:
        words = []
        for word_id in sentences_ids:
            word = t_voc[word_id]
            # if word == pad_id:
            #     break
            words.append(word)
    sentences.append(words)
    return sentences


if __name__ == '__main__':
    # for i in range(100):
    #     source_batch_pad,target_input_batch_pad,target_output_batch_pad,source_seq_length,target_seq_length = generate_batch(32)
    #     print(source_seq_length)
    # s_vec,t_vec = embed()
    # print
    batch_sentence_ids = [[745, 56, 743, 684, 508, 653, 743, 743, 208, 421, 105, 481, 433, 743, 743, 743, 508, 929, 929,930], [
        745, 56, 743, 684, 508, 653, 743, 743, 208, 421, 105, 481, 433, 743, 743, 743, 508, 929, 929]]
    print(len(batch_sentence_ids[0]))
    sentences = id2words_source(batch_sentence_ids)
    print(len(sentences[0][0]))
