# STA 142A FINAL PROJECT
# Toxic Comment Classification Challenge (Kaggle Competition Dataset)
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras, reduce_mean, square
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.keras.utils import losses_utils

import re
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import gensim
import logging

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# text preprocessing
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;\n]')
BAD_SYMBOLS_RE = re.compile('[^a-z ]')
STOPWORDS = set(stopwords.words('english'))

# function to clean the text by removing unnecessary symbols and useless stopwords
def clean_text(text):
    '''

    text: a string

    returns: a modified shorter string

    '''
    text = BeautifulSoup(text, "html.parser").text # html encoding
    text = text.lower() # convert text to all lowercase
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replaces the REPLACE_BY_SPACE_RE symbols with a space
    text = BAD_SYMBOLS_RE.sub('', text) # removes BAD_SYMBOLS_RE
    text =  ' '.join(word for word in text.split() if word not in STOPWORDS) # deletes stopwords
    return text # returns cleaner comment_text

# print(len(train['comment_text'][3645])) # lenght was 4424
# print(len(clean_text(train['comment_text'][3645]))) # after cleaning it the length was 2783
train['comment_text'] = train['comment_text'].apply(clean_text)

# The maximum number of words that will be used for tokenizing. (most frequent words)
MAX_NUM_WORDS = 50000
# Maximum number of words in each comment_text value
MAX_SEQ_LEN = 100

# tokenizing the comment_text values
TOKENIZER = text.Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
TOKENIZER.fit_on_texts(train['comment_text'].values)

word_index = TOKENIZER.word_index
print('Found %s unique tokens.' % len(word_index))

x_train = TOKENIZER.texts_to_sequences(train["comment_text"].values)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQ_LEN)

# 6 labels for classification ranging from toxic to identity hate
classes_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[classes_list].to_numpy()

test_labels = pd.read_csv('test_labels.csv')
test = test.merge(test_labels, on='id') # merging the test dataset with the test_labels ['toxic',....,'identity_hate']
test = test[test['toxic'] != -1] # removing all rows that have -1 for all values in the labels

test['comment_text'] = test['comment_text'].apply(clean_text)
x_test = TOKENIZER.texts_to_sequences(test['comment_text'].values)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQ_LEN)

y_test = test[classes_list].to_numpy()

# wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
wv = gensim.models.KeyedVectors.load_word2vec_format("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", binary=True)

# word 2 vec implementation
def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])

x_train_word_avg = word_averaging_list(wv, x_train)
x_test_word_avg = word_averaging_list(wv, x_test)

# loss function for model

# recurrent neural network model
rnn_model = keras.Sequential()
rnn_model.add(layers.Embedding(MAX_NUM_WORDS, 300, input_length=1)) # must be 300 b/c of google model
rnn_model.add(layers.Dropout(0.3))
rnn_model.add(layers.LSTM(50, dropout=0.2, recurrent_dropout=0.2))
rnn_model.add(layers.Dense(6, activation='sigmoid'))
rnn_model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC(multi_label=True)])

epochs = 2
batch_size = 512
history = rnn_model.fit(x_train_word_avg, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

print(rnn_model.evaluate(x_test_word_avg, y_test))
# loss function binary_crossentropy gave me 96.38 % accuracy
# loss function categorical_crossentropy gave me 96.32 % accuracy