#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use gensim to create word2vec
生成词向量
"""
from gensim.models import word2vec
import time
import os
import numpy as np

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
embedding_size = 300


def emb(filepath, modelpath, vocab, vec):
    sentences = word2vec.Text8Corpus(filepath)
    model = word2vec.Word2Vec(
        sentences, size=embedding_size, min_count=5, max_vocab_size=100000)
    model.save(modelpath)
    vocab_f = open(vocab, 'w+')
    vec_f = open(vec, 'w+')
    model = word2vec.Word2Vec.load(modelpath)
    all_words = set()
    for line in open(filepath):
        words = line.split(" ")
        for word in words:
            word = word.strip()
            if word != '':
                all_words.add(word)
    for word in all_words:
        try:
            vector = model[word]
            v = ' '.join(str(num) for num in vector)
            vec_f.writelines(v + '\n')
            vocab_f.writelines(word + '\n')
        except:
            pass
    random_vec = ' '.join(str(num) for num in (-1 + 2 *
                                               np.random.random(embedding_size)).astype(np.float32))

    vocab_f.write(unk + '\n')
    vec_f.write(random_vec + '\n')
    vocab_f.write(pad + '\n')
    vec_f.write(random_vec + '\n')
    vocab_f.write(tgt_sos + '\n')
    vec_f.write(random_vec + '\n')
    vocab_f.write(tgt_eos + '\n')
    vec_f.write(random_vec + '\n')
    vocab_f.close()
    vec_f.close()


if __name__ == '__main__':
    file_path = pwd + '/data/train.conll'
    vocab_path = pwd + '/data/vocab.txt'
    vec_path = pwd + '/data/vec.txt'
    model_path = pwd + '/data/source'
    unk = 'uuunnnkkk'
    pad = '<pad>'
    tgt_sos = '\<s>'
    tgt_eos = '\</s>'
    emb(file_path, model_path, vocab_path, vec_path)
