#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use jieba and gensim to create word2vec
生成词向量
"""
import jieba
from gensim.models import word2vec
import time
import os

dir_path = "/Users/luheng/dureader/data/preprocessed/trainset/"
file_path = dir_path + 'train.conll'



vocabfile = dir_path + 'vocab.txt'
word2vecfile = dir_path + 'vec.txt'
sentences = word2vec.Text8Corpus(file_path)
model = word2vec.Word2Vec(sentences, size=50, min_count=5,max_vocab_size=100000)
model.save(dir_path+'mymodel')
vocab = open(vocabfile, 'w+')
vec = open(word2vecfile, 'w+')

'''
这边需要把vocab改成唯一的
'''

model = word2vec.Word2Vec.load(dir_path+'mymodel')
all_words = set()

for line in open(file_path):
    words = line.split(" ")
    for word in words:
        word = word.strip()
        if word != '':
            all_words.add(word)
print(len(all_words))
for word in all_words:
    try:
        vector = model[word]
        v = ' '.join(str(num) for num in vector)
        vec.writelines(v + '\n')
        vocab.writelines(word + '\n')
    except:
        pass
vocab.close()
vec.close()
