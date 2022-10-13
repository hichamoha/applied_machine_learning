
import sys
import os

from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math

OPTIMIZER = 'rmsprop'
SCALER = True
SIMPLE_MODEL = False
BATCH_SIZE = 128
EPOCHS = 1
MINI_CORPUS = True
EMBEDDING_DIM = 100
UNKNOWN_TOKEN = '__UNK__'
W_SIZE = 2

## Preprocessing
# Preprocessing is more complex through 4 steps.

#### 1 - function for Reading the corpus

BASE_DIR = '/media/hi8826mo-s/BEEE-DE51/Ultimi/EDAN95_Applied_Machine_Learning/labs/lab4/'

def load_conll2009_pos():
    train_file = BASE_DIR + 'NER-data/eng.train'
    dev_file = BASE_DIR + 'NER-data/eng.valid'
    test_file = BASE_DIR + 'NER-data/eng.test'
    # test2_file = 'simple_pos_test.txt'

    #column_names = ['id', 'form', 'lemma', 'plemma', 'pos', 'ppos']
    column_names = ['form', 'ppos', 'pchunk', 'ner']

    """
    The strip() method returns a copy of the string in which all chars have been 
    stripped from the beginning and the end of the string 
    (default whitespace characters).
    """
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()


    return train_sentences, dev_sentences, test_sentences, column_names

# Read the corpus
train_sentences, dev_sentences, test_sentences, column_names = load_conll2009_pos()


#### 2 - class for tokenization / storing the rows in dictionaries
import regex as re

class Token(dict):
    pass

class CoNLLDictorizer:

    def __init__(self, column_names, sent_sep='\n\n', col_sep=' +'):
        self.column_names = column_names
        self.sent_sep = sent_sep
        self.col_sep = col_sep

    def fit(self):
        pass

    def transform(self, corpus):
        corpus = corpus.strip()
        sentences = re.split(self.sent_sep, corpus)
        return list(map(self._split_in_words, sentences))

    def fit_transform(self, corpus):
        return self.transform(corpus)

    def _split_in_words(self, sentence):
        rows = re.split('\n', sentence)
        return [Token(dict(zip(self.column_names,
                               re.split(self.col_sep, row))))
                for row in rows]



# store the rows in dictionaries
conll_dict = CoNLLDictorizer(column_names, col_sep=' +')

train_dict = conll_dict.transform(train_sentences)

#if MINI_CORPUS:
#    train_dict = train_dict[:len(train_dict) // 5]

test_dict = conll_dict.transform(test_sentences)

print('First sentence, train: \n', train_dict[0])
print('Second sentence, train: \n', train_dict[1])



#### 3 Extract the features and store them in dictionaries

class ContextDictorizer():
    """
    Extract contexts of words in a sequence
    Contexts are of w_size to the left and to the right
    Builds an X matrix in the form of a dictionary
    and possibly extracts the output, y, if not in the test step
    If the test_step is True, returns y = []
    """

    #def __init__(self, input='form', output='pos', w_size=2, tolower=True):
    def __init__(self, input='form', output='ner', w_size=2, tolower=True):
        self.BOS_symbol = '__BOS__'
        self.EOS_symbol = '__EOS__'
        self.input = input
        self.output = output
        self.w_size = w_size
        self.tolower = tolower
        # This was not correct as the names were not sorted
        # self.feature_names = [input + '_' + str(i)
        #                     for i in range(-w_size, w_size + 1)]
        # To be sure the names are ordered
        zeros = math.ceil(math.log10(2 * w_size + 1))
        self.feature_names = [input + '_' + str(i).zfill(zeros) for
                              i in range(2 * w_size + 1)]

    def fit(self, sentences):
        """
        Build the padding rows
        :param sentences:
        :return:
        """
        self.column_names = sentences[0][0].keys()
        start = [self.BOS_symbol] * len(self.column_names)
        end = [self.EOS_symbol] * len(self.column_names)
        start_token = Token(dict(zip(self.column_names, start)))
        end_token = Token(dict(zip(self.column_names, end)))
        self.start_rows = [start_token] * self.w_size
        self.end_rows = [end_token] * self.w_size

    def transform(self, sentences, training_step=True):
        X_corpus = []
        y_corpus = []
        for sentence in sentences:
            X, y = self._transform_sentence(sentence, training_step)
            X_corpus += X
            if training_step:
                y_corpus += y
        return X_corpus, y_corpus

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)

    def _transform_sentence(self, sentence, training_step=True):
        # We extract y
        if training_step:
            y = [row[self.output] for row in sentence]    # XXXXXX
        else:
            y = None

        # We pad the sentence
        sentence = self.start_rows + sentence + self.end_rows

        # We extract the features
        X = list()
        for i in range(len(sentence) - 2 * self.w_size):
            # x is a row of X
            x = list()
            # The words in lower case
            for j in range(2 * self.w_size + 1):
                if self.tolower:
                    x.append(sentence[i + j][self.input].lower())
                else:
                    x.append(sentence[i + j][self.input])
            # We represent the feature vector as a dictionary
            X.append(dict(zip(self.feature_names, x)))
        return X, y

    def print_example(self, sentences, id=1968):
        """
        :param corpus:
        :param id:
        :return:
        """
        # We print the features to check they match Table 8.1 in my book (second edition)
        # We use the training step extraction with the dynamic features
        Xs, ys = self._transform_sentence(sentences[id])
        print('X for sentence #', id, Xs)
        print('\n')
        print('y for sentence #', id, ys)
        print('\n')

###### Running the Feature Extraction
context_dictorizer = ContextDictorizer()
context_dictorizer.fit(train_dict)
x_dict, y_cat = context_dictorizer.transform(train_dict)  # XXXXXX keyError 'pos'

context_dictorizer.print_example(train_dict)



#### 4 - Vectorize the symbols
# We transform the x symbols into numbers
dict_vectorizer = DictVectorizer()
x_num = dict_vectorizer.fit_transform(x_dict)
