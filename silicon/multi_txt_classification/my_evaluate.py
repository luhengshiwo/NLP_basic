#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import numpy as np
import pandas as pd
from sklearn import metrics
out = pd.read_csv('data/out.csv',sep='\t',header=None)
test = pd.read_csv('data/test.csv',sep='\t',header=None)
predicts = out.iloc[:,1]
y = test.iloc[:,3]
precision = metrics.precision_score(y, predicts)
recall = metrics.recall_score(y, predicts)
f1 = metrics.f1_score(y, predicts)
print('precision:{:.5}'.format(precision)+'\trecall:{:.5}'.format(recall)+'\tf1:{:.5}'.format(f1))