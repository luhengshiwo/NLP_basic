#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
import numpy as np
import jieba
'''
1，生成一个一个的batch
2，将每个sentence转化为id
3，加上unk，start，end
4，pad
5，返回长度
'''

pwd = os.getcwd()
# pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
# source_path = pwd + '/data/cut1000.zh'
# target_path = pwd + '/data/small1000.en'
vocab_path = pwd + '/data/vocab.txt'
vec_path = pwd + '/data/vec.txt'
mydict = pwd+'/data/dict.txt'
# dev_source_path = pwd + '/data/cutdev128.zh'
# dev_target_path = pwd + '/data/dev128.en'
unk = 'uuunnnkkk'
pad_id = '<pad>'

voc = []
vec = []


def vocab_parse(path, vocab):
    for line in open(path):
        vocab.append(line.strip())


def vec_parse(path, vec):
    for line in open(path):
        vector = line.strip()
        vec.append(list(map(float, vector.split())))


vocab_parse(vocab_path, voc)
vec_parse(vec_path, vec)


def find_index(vocab, word):
    if word in vocab:
        return vocab.index(word)
    else:
        return vocab.index(unk)


def sentence_token(sentence):
    words = sentence.strip().split(' ')
    newsentence = []
    vocab = voc
    for word in words:
        newsentence.append(find_index(vocab, word))
    return newsentence


def pad_sentence_batch(sentences):
    pad_int = find_index(voc, pad_id)
    max_sentence = max([len(sentence) for sentence in sentences])
    newsentences = []
    sentence_length = []
    for sentence in sentences:
        newsentences.append(sentence + [pad_int]
                            * (max_sentence - len(sentence)))
        sentence_length.append(len(sentence))
    return (newsentences, sentence_length)


def shuffle_aligned_list(data1, data2, data3):
    num = len(data1)
    p = np.random.permutation(num)
    return ([data1[i] for i in p], [data2[i] for i in p], [data3[i] for i in p])


def batch_generator(data1, data2, data3, batch_size, shuffle):
    if shuffle:
        data1, data2, data3 = shuffle_aligned_list(data1, data2, data3)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data1):
            batch_count = 0
            if shuffle:
                data1, data2, data3 = shuffle_aligned_list(data1, data2, data3)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield (data1[start:end], data2[start:end], data3[start:end])


def generate_batch(source_path, target_path, simi_path, batch_size, shuffle=True):
    source_file = open(source_path)
    target_file = open(target_path)
    simi_file = open(simi_path)
    encoder_inputs = []
    decoder_inputs = []
    simi = []
    for line in source_file:
        sentence = sentence_token(line)
        encoder_inputs.append(sentence)
    for line in target_file:
        sentence_input = sentence_token(line)
        decoder_inputs.append(sentence_input)
    for line in simi_file:
        simi_input = int(line.strip())
        simi.append(simi_input)
    source_file.close()
    target_file.close()
    simi_file.close()
    gen = batch_generator(encoder_inputs, decoder_inputs,
                          simi, batch_size, shuffle=shuffle)
    source_input_batch, target_input_batch, simi_input_batch = next(gen)
    source_batch_pad, source_seq_length = pad_sentence_batch(
        source_input_batch)
    target_input_batch_pad, target_seq_length = pad_sentence_batch(
        target_input_batch)
    return source_batch_pad, target_input_batch_pad, simi_input_batch, source_seq_length, target_seq_length


def embed():
    return(vec, voc)


def id2words(batch_sentence_ids):
    sentences = []
    # print(voc)
    for sentences_ids in batch_sentence_ids:
        words = []
        for word_id in sentences_ids:
            word = voc[word_id]
            if word == pad_id:
                break
            words.append(word)
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences


def listbatch(onelist, batch_size):
    return np.tile(onelist, (batch_size, 1))
def intbatch(one, batch_size):
    return [one for i in range(batch_size)]

def evaluate(line, batch_size):
    jieba.load_userdict(mydict)
    line = line.replace('***','num')
    lines = line.strip().split('\t')
    ids = lines[0]
    source_jieba = " ".join(jieba.cut(lines[1]))
    target_jieba = " ".join(jieba.cut(lines[2]))
    source_index = sentence_token(source_jieba)
    target_index = sentence_token(target_jieba)
    source_seq_length = len(source_index)
    target_seq_length = len(target_index)
    return (ids, listbatch(source_index, batch_size), listbatch(target_index, batch_size), intbatch(source_seq_length, batch_size), intbatch(target_seq_length, batch_size))


if __name__ == '__main__':
    source = pwd + '/data/sourcedata.conll'
    target = pwd + '/data/targetdata.conll'
    simi = pwd + '/data/simi.conll'
    # for i in range(2):
    #     source_batch_pad,target_input_batch_pad,simi_input_batch,source_seq_length,target_seq_length = generate_batch(source,target,simi,128)
    #     print(np.shape(source_batch_pad))
    # s_vec,t_vec = embed()
    # print
    # batch_sentence_ids = [[745, 56, 743, 684, 508, 653, 743, 743, 208, 421, 105, 481, 433, 743, 743, 743, 508, 743, 213, 743, 743, 735, 181, 579, 743, 508, 743, 213, 481, 237, 735, 78, 213, 508, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 743, 213, 508, 743, 743, 237, 455, 743, 242, 686, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 302, 743, 449, 572, 268, 84, 508, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 562, 239, 743, 12, 508, 743, 476, 743, 661, 432, 693, 743, 84, 80, 214, 743, 307, 743, 306, 508, 743, 239, 290, 213, 611, 66, 743, 239, 743, 12, 743, 743, 130, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 571, 587, 508, 57, 743, 743, 743, 743, 450, 80, 61, 213, 743, 743, 170, 540, 562, 239, 743, 12, 508, 743, 743, 670, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 659, 84, 743, 237, 743, 743, 12, 80, 743, 213, 508, 394, 213, 80, 70, 213, 0, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 743, 508, 743, 743, 384, 213, 743, 743, 480, 743, 743, 237, 299, 79, 239, 743, 126, 102, 390, 697, 450, 168, 554, 685, 653, 478, 108, 129, 665, 133, 478, 743, 743, 743, 743, 211, 654, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 68, 743, 84, 743, 450, 743, 3, 743, 80, 743, 743, 743, 559, 280, 743, 197, 80, 81, 213, 743, 743, 239, 261, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 239, 387, 307, 90, 69, 592, 53, 306, 743, 508, 71, 743, 208, 743, 237, 714, 78, 213, 508, 699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 401, 25, 684, 743, 306, 531, 183, 474, 661, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 437, 739, 20, 222, 5, 743, 446, 217, 12, 25, 213, 508, 476, 743, 239, 743, 743, 743, 213, 508, 99, 695, 213, 119, 84, 508, 653, 458, 66, 704, 449, 306, 743, 508, 706, 239, 560, 213, 80, 743, 166, 743, 213, 3, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 127, 561, 743, 743, 478, 508, 156, 237, 565, 237, 593, 561, 222, 743, 743, 213, 333, 38, 30, 12, 577, 536, 720, 508, 99, 46, 743, 633, 661, 671, 743, 561, 222, 743, 743, 213, 333, 711, 55, 577, 743, 508, 646, 213, 301, 743, 743, 14, 539, 239, 561, 222, 743, 743, 213, 743, 711, 55, 12, 577, 536, 743, 508, 646, 213, 743], [745, 626, 307, 84, 743, 508, 11, 264, 660, 313, 213, 743, 235, 239, 450, 660, 279, 213, 743, 675, 239, 660, 743, 743, 213, 372, 273, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 626, 743, 743, 213, 317, 743, 433, 743, 12, 561, 701, 743, 743, 84, 133, 406, 237, 508, 548, 213, 547, 703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 598, 388, 237, 508, 743, 743, 213, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [
    #     745, 48, 538, 743, 508, 383, 743, 237, 743, 580, 377, 306, 687, 461, 84, 97, 237, 80, 743, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 305, 315, 306, 743, 627, 743, 214, 680, 743, 122, 743, 508, 715, 743, 496, 698, 549, 128, 478, 214, 523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 743, 561, 57, 166, 743, 237, 38, 30, 328, 743, 561, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 91, 84, 80, 743, 743, 743, 743, 293, 743, 264, 80, 255, 743, 328, 102, 80, 743, 205, 743, 661, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 213, 154, 90, 444, 434, 213, 307, 743, 66, 306, 743, 508, 743, 743, 577, 132, 660, 94, 743, 237, 743, 508, 567, 208, 80, 378, 579, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 743, 66, 579, 686, 743, 306, 597, 594, 345, 418, 237, 508, 699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 508, 743, 582, 661, 743, 404, 743, 306, 743, 508, 743, 213, 508, 743, 287, 239, 306, 87, 743, 743, 237, 565, 239, 237, 3, 743, 743, 388, 164, 508, 99, 46, 239, 508, 743, 433, 112, 433, 450, 400, 213, 508, 546, 209, 508, 743, 213, 508, 743, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 736, 648, 234, 89, 237, 743, 119, 576, 185, 306, 590, 541, 743, 149, 12, 541, 743, 239, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 202, 412, 239, 743, 213, 524, 743, 219, 18, 450, 492, 237, 743, 637, 743, 208, 743, 743, 239, 92, 511, 307, 502, 239, 234, 89, 356, 208, 242, 743, 501, 508, 743, 530, 213, 82, 606, 242, 743, 686, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 258, 401, 743, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 305, 743, 328, 80, 743, 159, 700, 84, 743, 661, 80, 727, 743, 306, 743, 80, 700, 743, 267, 83, 743, 84, 436, 306, 65, 508, 486, 743, 436, 306, 434, 508, 66, 478, 660, 166, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 132, 2, 478, 743, 38, 743, 433, 80, 743, 213, 508, 743, 743, 425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 272, 22, 213, 508, 162, 132, 306, 21, 545, 450, 554, 478, 508, 683, 743, 12, 311, 107, 630, 239, 306, 743, 121, 237, 577, 596, 69, 686, 128, 237, 472, 340, 517, 126, 239, 736, 213, 508, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 687, 743, 242, 686, 254, 237, 743, 48, 743, 743, 637, 239, 508, 743, 213, 547, 239, 735, 234, 89, 66, 237, 505, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 743, 717, 478, 12, 508, 252, 213, 508, 99, 695, 213, 119, 661, 321, 237, 743, 306, 80, 743, 206, 110, 743, 320, 743, 561, 743, 743, 743, 239, 743, 743, 239, 743, 306, 743, 743, 239, 743, 213, 743, 239, 735, 743, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [745, 626, 743, 743, 306, 508, 743, 743, 684, 80, 743, 307, 743, 743, 239, 743, 577, 576, 2, 18, 237, 243, 30, 743, 386, 743, 18, 450, 508, 743, 458, 234, 89, 425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # sentences = id2words(batch_sentence_ids)
    # print(sentences)
    print(evaluate("58	支付宝花呗分期还款怎么提前还清	蚂蚁花呗分期付款可以全部提前还款吗	1", 12))
