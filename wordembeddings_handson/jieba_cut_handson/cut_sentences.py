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
filepath = father_path+'/data/dev128.zh'
write_path = father_path+'/data/cutdev128.zh'
out_file = open(write_path, 'w+')
f = open(filepath)
for line in f:
    line = line.strip('\n')
    seg_list = jieba.cut(line)
    newline = " ".join(seg_list)
    out_file.write(newline+'\n')
f.close()
out_file.close()