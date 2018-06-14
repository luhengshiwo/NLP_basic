#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import time
import numpy as np
import jieba
import os
import subprocess
from gensim.models import word2vec
from parameters import Para

embedding_size = Para.embedding_size
unk = Para.unk
pad = Para.pad
tgt_sos = Para.tgt_sos
tgt_eos = Para.tgt_eos
origin_path = 'data/origin.csv'
cut_path = 'data/cut.csv'
mydict = 'data/dict.txt'
file_path = 'data/train.csv'
vocab_path = 'data/vocab.txt'
vec_path = 'data/vec.txt'
model_path = 'data/source'

def jiaba_cut(origin_path,cut_path):
    train_file = open(origin_path)
    cut_file = open(cut_path,'w+')
    jieba.load_userdict(mydict)
    for line in train_file:
        lines = line.strip().split('\t')
        ids = lines[0]
        first = " ".join(jieba.cut(lines[1]))
        second = " ".join(jieba.cut(lines[2]))
        simi = lines[3]
        cut_file.write(ids+'\t'+first+'\t'+second+'\t'+simi+'\n')
    train_file.close()
    cut_file.close()

def bash_shell():
    shell_path = 'train_dev_test_split.sh'
    subprocess.call(['bash',shell_path])

def create_embeddings(filepath, modelpath, vocab, vec):
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
    os.remove(modelpath)

def main():
    jiaba_cut(origin_path,cut_path)
    bash_shell()
    create_embeddings(file_path, model_path, vocab_path, vec_path)

if __name__ == '__main__':
    tic = time.time()
    main()
    tok = time.time()
    cost = tok-tic
    print('cost time:{:.2f}'.format(cost))