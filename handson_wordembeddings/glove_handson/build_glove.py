# coding: utf-8


from gensim import utils, corpora, matutils, models
import glove
import pickle
import time
import numpy as np

print('Building model...')
start_time = time.time()
sentences = models.word2vec.LineSentence('train.conll')

id2word = corpora.Dictionary(sentences)
id2word.filter_extremes(no_below=1, no_above=1, keep_n=30000)

tokens = [word for word in id2word.values()]
word2id = dict((word, id) for id, word in id2word.iteritems())

filter_text = lambda text: [word for word in text if word in word2id]
filtered_text = lambda: (filter_text(text) for text in sentences) 

cooccur = glove.Corpus(dictionary=word2id)
cooccur.fit(filtered_text(), window=5)
model_glove = glove.Glove(no_components=300, learning_rate=0.05)
print('model has been build, time consumed {}'.format(time.time() - start_time))

print('Start training...')
start_time = time.time()
model_glove.fit(cooccur.matrix, epochs=10, verbose=True, no_threads=8)
print('Finished training, time consumed {}'.format(time.time() - start_time))


model_glove.add_dictionary(cooccur.dictionary)
model_glove.most_similar("打开")

print('Saving model')
np.save('word_vectors', model_glove.word_vectors)
with open('word2id.dict', 'wb') as f:
    pickle.dump(model_glove.dictionary, f)
