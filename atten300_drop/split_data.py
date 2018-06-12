#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import logging
import json
import numpy as np
import jieba

pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
# grader_father=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
# print(father_path)
filepath = pwd+'/data/train.csv'
source_path = pwd+'/data/train1.conll'
target_path = pwd+'/data/train2.conll'
simi_path = pwd+'/data/train_simi.conll'
# filepath = pwd+'/data/dev.csv'
# source_path = pwd+'/data/dev1.conll'
# target_path = pwd+'/data/dev2.conll'
# simi_path = pwd+'/data/dev_simi.conll'
source_file = open(source_path, 'w+')
target_file = open(target_path, 'w+')
simi_file = open(simi_path, 'w+')
f = open(filepath)
mydict = pwd+'/data/dict.txt'
for line in f:
    jieba.load_userdict(mydict)
    line = line.replace('***','num')
    line = line.strip('\n').split('\t')
    seg_list1 = jieba.cut(line[1])
    seg_list2 = jieba.cut(line[2])
    newline1 = " ".join(seg_list1)
    newline2 = " ".join(seg_list2)
    source_file.write(newline1+'\n')
    target_file.write(newline2+'\n')
    simi_file.write(line[3]+'\n')
f.close()
source_file.close()
target_file.close()
simi_file.close()
# wordvector_file = pwd+'/data/train.conll'
# wordvector = open(wordvector_file, 'w+')
# for line in f:
#     jieba.load_userdict(mydict)
#     line = line.replace('***','num')
#     line = line.strip('\n').split('\t')
#     seg_list1 = jieba.cut(line[1])
#     seg_list2 = jieba.cut(line[2])
#     newline1 = " ".join(seg_list1)
#     newline2 = " ".join(seg_list2)
#     wordvector.write(newline1+'\n')
#     wordvector.write(newline2+'\n')
# wordvector.close()
