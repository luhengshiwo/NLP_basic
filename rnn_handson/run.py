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
embedding_size = 50


def emb(filepath, modelpath, vocab, vec, is_source=True):
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
    if is_source:
        vocab_f.write(unk + '\n')
        vec_f.write(random_vec + '\n')
        vocab_f.write(pad + '\n')
        vec_f.write(random_vec + '\n')
    else:
        vocab_f.write(unk + '\n')
        vocab_f.write(tgt_sos_id + '\n')
        vocab_f.write(tgt_eos_id + '\n')
        vocab_f.write(pad+'\n')
        vec_f.write(random_vec + '\n')
        vec_f.write(random_vec + '\n')
        vec_f.write(random_vec + '\n')
        vec_f.write(random_vec + '\n')
    vocab_f.close()
    vec_f.close()


if __name__ == '__main__':
    source_path = father_path + '/data/cut1000.zh'
    target_path = father_path + '/data/small1000.en'
    source_vocab = father_path + '/data/source_vocab.txt'
    source_vec = father_path + '/data/source_vec.txt'
    target_vocab = father_path + '/data/target_vocab.txt'
    target_vec = father_path + '/data/target_vec.txt'
    source_model = father_path + '/data/source'
    target_model = father_path + '/data/target'

    unk = 'uuunnnkkk'
    pad = '<pad>'
    tgt_sos_id = '\<s>'
    tgt_eos_id = '\</s>'
    emb(source_path, source_model, source_vocab, source_vec)
    emb(target_path, target_model, target_vocab, target_vec, is_source=False)
