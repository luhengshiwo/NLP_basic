from scipy.io import wavfile
import os
import math
from scipy import signal
import numpy as np
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
rootdir = pwd+'/wavdata'
lists = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(lists)):
    path = os.path.join(rootdir,lists[i])
    if os.path.isfile(path):
        rate, wavdata = wavfile.read(path)
        a, b, data = signal.spectrogram(wavdata, rate)
        newdata = np.transpose(data)
        print(newdata.shape)
