import os
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

import pickle as pickle




class IMDBDataset(object):
    def __init__(self, path='sst-2', max_vocab_size=None):
        self.path = path
        self.train_path = path + '/train.tsv'
        self.valid_path = path + '/valid.tsv'
        self.test_path = path + '/test.tsv'

        self.max_vocab_size = max_vocab_size

        train_text, self.train_y = self.read_text(self.train_path)
        valid_text, self.valid_y = self.read_text(self.valid_path)
        test_text, self.test_y = self.read_text(self.test_path)
        self.train_text = train_text
        self.valid_text = valid_text
        self.test_text = test_text
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        # nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        # train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        # test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(self.train_text)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        # sorted_words = sorted([x for x in self.tokenizer.word_counts])
        # self.top_words = sorted_words[:max_vocab_size-1]
        # self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.valid_seqs = self.tokenizer.texts_to_sequences(self.valid_text)
        self.valid_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.valid_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_valid_set.pickle', 'wb') as f:
            pickle.dump((self.valid_text, self.valid_seqs, self.valid_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)


    def read_text(self, filename):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        neg_list = []
        pos_num=0
        neg_num=0
        with open(filename) as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.split('\t')
                if (int(line[1])-int('0'))==0:
                    neg_list.append(line[0])
                    neg_num += 1
                else:
                    pos_list.append(line[0])
                    pos_num += 1
        data_list = pos_list + neg_list
        labels_list = [1] * pos_num + [0] * neg_num
        return data_list, labels_list

    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)
