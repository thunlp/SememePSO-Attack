from __future__ import print_function
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import keras
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import pickle

import numpy as np
np.random.seed(1337)  # for reproducibility


def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()


def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(
            data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(
            data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)


def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    print(max(len(x.split()) for x in left))
    print(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y


training = get_data('snli_data/snli_1.0/snli_1.0_train.jsonl')
validation = get_data('snli_data/snli_1.0/snli_1.0_dev.jsonl')
test = get_data('snli_data/snli_1.0/snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def to_seq(X): return pad_sequences(
    tokenizer.texts_to_sequences(X), maxlen=100,padding='post')


def prepare_data(data): return (to_seq(data[0]), to_seq(data[1]), data[2])

if __name__ == '__main__':
    training = prepare_data(training)
    validation = prepare_data(validation)
    test = prepare_data(test)

    with open('nli_tokenizer.pkl', 'wb') as fh:
        pickle.dump(tokenizer, fh)
    with open('nli_seqs.pkl', 'wb') as fh:
        pickle.dump((training,validation,test), fh)
