#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import logging
import json
import numpy as np
import jieba

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
grader_father=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
# print(father_path)
filepath = pwd+'/data/atec_nlp_sim_train.csv'
write_path = pwd+'/data/traindata.conll'
out_file = open(write_path, 'w+')
f = open(filepath)
for line in f:
    line = line.strip('\n').split('\t')
    seg_list1 = jieba.cut(line[1])
    seg_list2 = jieba.cut(line[2])
    newline1 = " ".join(seg_list1)
    newline2 = " ".join(seg_list2)
    out_file.write(newline1+'\n')
    out_file.write(newline2+'\n')
f.close()
out_file.close()