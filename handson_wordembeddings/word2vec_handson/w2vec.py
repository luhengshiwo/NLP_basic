import json
import numpy as np
# file_path = 'search.dev.json'

write_path = "/Users/luheng/dureader/data/preprocessed/trainset/train.conll"
write_file = open(write_path,'w+')
def myword2vec(file_path):
    file = open(file_path)
    for line in file:
        line_dict = json.loads(line)
        for doc in line_dict["documents"]:
            write_file.write(' '.join(doc["segmented_title"])+'\n')
            for doc_single in doc["segmented_paragraphs"]:
                write_file.write(' '.join(doc_single)+'\n')
        write_file.write(' '.join(doc["segmented_title"])+'\n')
        for doc_small in line_dict["segmented_answers"]:
            write_file.write(' '.join(doc_small)+'\n')

file_path1 = "/Users/luheng/dureader/data/preprocessed/trainset/search.train.json"
file_path2 = "/Users/luheng/dureader/data/preprocessed/trainset/zhidao.train.json"
# file_path3 = "/Users/luheng/dureader/data/preprocessed/devset/zhidao.dev.json"
myword2vec(file_path1)
myword2vec(file_path2)
# myword2vec(file_path3)
write_file.write('<s>' + '\n')
write_file.write('</s>' + '\n')
write_file.write('NNNUUUMMM' + '\n')
write_file.write('NNNKKK' + '\n')
write_file.write('KKKGGG' + '\n')
# print(np.random.rand(50))