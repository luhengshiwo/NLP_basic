#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import time
import os
import numpy as np
from parameters import Para
from gensim.models import word2vec
import jieba
'''
1，生成一个一个的batch
2，将每个sentence转化为id
3，加上unk，start，end
4，pad
5，返回长度
'''

# pwd = os.getcwd()
# pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
mydict = 'data/dict.txt'
source_path = 'data/train.csv'
vocab_path = 'data/vocab.txt'
vec_path = 'data/vec.txt'
unk = Para.unk
pad = Para.pad
batch_size = Para.batch_size
voc = []
vec = []
embedding_size = Para.embedding_size

def vocab_parse(path, vocab):
    for line in open(path):
        vocab.append(line.strip())


def vec_parse(path, vec):
    for line in open(path):
        vector = line.strip()
        vec.append(list(map(float, vector.split())))


vocab_parse(vocab_path, voc)
vec_parse(vec_path, vec)
def embeddings():
    return(vec, voc)

def find_index(vocab, word):
    if word in vocab:
        return vocab.index(word)
    else:
        return vocab.index(unk)


def sentence_token(sentence):
    words = sentence.strip().split(' ')
    newsentence = []
    vocab = voc
    for word in words:
        newsentence.append(find_index(vocab, word))
    return newsentence


def pad_sentence_batch(sentences):
    pad_int = find_index(voc, pad)
    max_sentence = max([len(sentence) for sentence in sentences])
    newsentences = []
    sentence_length = []
    for sentence in sentences:
        newsentences.append(sentence + [pad_int]
                            * (max_sentence - len(sentence)))
        sentence_length.append(len(sentence))
    return (newsentences, sentence_length)


def shuffle_aligned_list(data0,data1, data2, data3):
    num = len(data1)
    p = np.random.permutation(num)
    return ([data0[i] for i in p], [data1[i] for i in p], [data2[i] for i in p], [data3[i] for i in p])

def generate_batch(source_path, batch_size, is_train=True, shuffle=False):
    source_file = open(source_path)
    ids = []
    first = []
    second = []
    simi = []
    for line in source_file:
        lines = line.strip().split('\t')
        first.append(sentence_token(lines[1]))
        second.append(sentence_token(lines[2]))
        if is_train:
            simi.append(int(lines[3]))
        else:
            ids.append(lines[0])
    source_file.close()
    if shuffle:
        ids,first, second, simi = shuffle_aligned_list(
            ids,first, second, simi)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(first):
            batch_count = 0
            if shuffle:
                ids,first, second, simi = shuffle_aligned_list(
                    ids,first, second, simi)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        first_batch_pad, first_seq_length = pad_sentence_batch(
            first[start:end])
        second_batch_pad, second_seq_length = pad_sentence_batch(
            second[start:end])
        if is_train:
            simi_batch = simi[start:end]
            yield (first_batch_pad, first_seq_length,second_batch_pad, second_seq_length,simi_batch)
        else:
            ids_batch = ids[start:end]
            yield (ids_batch,first_batch_pad, first_seq_length,second_batch_pad, second_seq_length)

def generate_batch_evaluate(source_path, batch_size=batch_size):
    source_file = open(source_path)
    ids = []
    first = []
    second = []
    for line in source_file:
        lines = line.strip().split('\t')
        for _ in range(batch_size):
            first.append(sentence_token(lines[1]))
            second.append(sentence_token(lines[2]))
            ids.append(lines[0]) 
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(first):
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        first_batch_pad, first_seq_length = pad_sentence_batch(
            first[start:end])
        second_batch_pad, second_seq_length = pad_sentence_batch(
            second[start:end])
        ids_batch = ids[start:end]
        yield (ids_batch,first_batch_pad, first_seq_length,second_batch_pad, second_seq_length)

def generate_batch_shell(source_sentence, batch_size=batch_size):
    ids = []
    first = []
    second = []
    simi = []
    lines = source_sentence.strip().split('__user__')
    jieba.load_userdict(mydict)
    for _ in range(batch_size):
        newline0 = " ".join(jieba.cut(lines[0]))
        newline1 = " ".join(jieba.cut(lines[0]))
        first.append(sentence_token(newline0))
        second.append(sentence_token(newline1))
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(first):
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        first_batch_pad, first_seq_length = pad_sentence_batch(
            first[start:end])
        second_batch_pad, second_seq_length = pad_sentence_batch(
            second[start:end])
        yield (first_batch_pad, first_seq_length,second_batch_pad, second_seq_length)

if __name__ == '__main__':
    gen = generate_batch(source_path,512,shuffle=False)
    tic = time.time()
    for i in range(1000): 
        a,b,c,d,e = next(gen)
    toc = time.time()
    print(toc-tic)