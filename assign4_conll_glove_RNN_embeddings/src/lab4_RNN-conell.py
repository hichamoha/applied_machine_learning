
                #### lab4_RNN-conell.py

# Python Headers - The modules
import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense

# Some Parameters
OPTIMIZER = 'rmsprop'
SCALER = True
SIMPLE_MODEL = False
BATCH_SIZE = 32
EPOCHS = 10

EMBEDDING_DIM = 100
UNKNOWN_TOKEN = '__UNK__'
W_SIZE = 2
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 150

LSTM_UNITS = 512

print(' \n XXXX READING ..... \n')


# Download the GloVe word embeddings
def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return:
    """
    file = file
    embeddings = {}
    
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    
    embeddings_dict = embeddings
    #embedded_words = sorted(list(embeddings_dict.keys()))
    
    return embeddings_dict

# Embbeddings dictionary
BASE_DIR = '/media/hi8826mo-s/BEEE-DE51/Ultimi/EDAN95_Applied_Machine_Learning/labs/lab4/'
embedding_file = BASE_DIR + 'glove.6B/glove.6B.100d.txt'
embeddings_dict = load(embedding_file)
embedded_words = sorted(list(embeddings_dict.keys()))

# Embeddings Index
BASE_DIR = '/media/hi8826mo-s/BEEE-DE51/Ultimi/EDAN95_Applied_Machine_Learning/labs/lab4/'
glove_dir = BASE_DIR + 'glove.6B/'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Using a cosine similarity, compute the 5 closest words to the words table, france, and sweden.
from scipy.spatial.distance import cosine

table = embeddings_dict['france']
#table = np.random.rand(1,100)
#a = np.random.rand(5,100)
#similarities = []
sim_dict = {}
#simmi = {}
for word, vector in embeddings_dict.items():
#for word in embeddings_dict.values:
        #print(cosine(table, a[i]))
        #print(cosine(table, a[i]))
        sim = cosine(table, vector)
        #sim = cosine_similarity(table,word)
        #key = embeddings_dict.get(word)
        #print (sim)
        #sim_dict.update(word=sim)
        sim_dict[word] = sim

sorted_by_value = sorted(sim_dict.items(), key = lambda kv: kv[1])        

sorted_by_value[0 : 6]
#print(table)
#print(len(sim_dict))

#print(similarities[:3])
#print(sim_dict.get(1.438331514596939))
#sim_dict.items()

# #####################   Preprocessing

# 1 - Loading the Corpus: function for reading the corpus

BASE_DIR = '/media/hi8826mo-s/BEEE-DE51/Ultimi/EDAN95_Applied_Machine_Learning/labs/lab4/'

def load_conll2009():
    train_file = BASE_DIR + 'NER-data/eng.train'
    dev_file = BASE_DIR + 'NER-data/eng.valid'
    test_file = BASE_DIR + 'NER-data/eng.test'
    # test2_file = 'simple_pos_test.txt'

    #column_names = ['id', 'form', 'lemma', 'plemma', 'pos', 'ppos']
    column_names = ['form', 'pos', 'chunk', 'ner']

    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    # test2_sentences = open(test2_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

# Read the corpus
train_sentences, dev_sentences, test_sentences, column_names = load_conll2009()

print(type(test_sentences))

#2 - Class for tokenization / Storing the rows in dictionaries

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

#conll_dict = CoNLLDictorizer(column_names, col_sep='\t')
conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
train_dict = conll_dict.transform(train_sentences)

#if MINI_CORPUS:
#   train_dict = train_dict[:len(train_dict) // 5]
    
test_dict = conll_dict.transform(test_sentences)
dev_dict = conll_dict.transform(dev_sentences)

print('First sentence, train: \n', train_dict[0])
print('Second sentence, train: \n', train_dict[1])
print('First sentence, test: \n', test_dict[0])

# 3 - Vectorizing the symbols (Creating the X and Y Sequencess)

# Function to build the two-way sequences: Two vectors: x and y

#def build_sequences(corpus_dict, key_x='form', key_y='upos', tolower=True):
def build_sequences(corpus_dict, key_x='form', key_y='ner', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = []
        y = []
        
        for word in sentence:
            x += [word[key_x]]
            y += [word[key_y]]
            
        if tolower:
            x = list(map(str.lower, x))
            
        X += [x]
        Y += [y]
    return X, Y

X_train_cat, Y_train_cat = build_sequences(train_dict)

#print('First sentence, words \n', X_train_cat[1])
#print('First sentence, NER \n', Y_train_cat[1])
#print('\n')

# Extracting the Unique Words and Named Entities Recognition

vocabulary_words = sorted(list(
    set([word for sentence 
         in X_train_cat for word in sentence])))

ner = sorted(list(set([ner for sentence 
                       in Y_train_cat for ner in sentence])))
print('Unique words in training \n', ner)
NB_CLASSES = len(ner)
print('\n')

#We create the dictionary
#We add two words for the padding symbol and unknown words

embeddings_words = embeddings_dict.keys()
print('Words in GloVe:',  len(embeddings_dict.keys()))

vocabulary_words = sorted(list(set(vocabulary_words + 
                                   list(embeddings_words))))
cnt_uniq = len(vocabulary_words) + 2
print('# unique words in the training vocabulary: embeddings and corpus:', 
      cnt_uniq)
print('\n')

# Function to convert the words or NER to indices

def to_index(X, idx):
    """
    Convert the word lists (or NER lists) to indexes
    :param X: List of word (or NER) lists
    :param idx: word to number dictionary
    :return:
    """
    X_idx = []
    for x in X:
        # We map the unknown words to one
        x_idx = list(map(lambda x: idx.get(x, 1), x))
        X_idx += [x_idx]
        
    return X_idx

# We create the indexes

# We start at one to make provision for the padding symbol 0 
# in RNN and LSTMs and 1 for the unknown words

rev_word_idx = dict(enumerate(vocabulary_words, start=2))
#rev_ner_idx = dict(enumerate(ner, start=2))
ner_rev_idx = dict(enumerate(ner, start=2))

word_idx = {v: k for k, v in rev_word_idx.items()}
#ner_idx = {v: k for k, v in rev_ner_idx.items()}
ner_idx = {v: k for k, v in ner_rev_idx.items()}

#print('word index: \n', list(word_idx.items())[:10])
#print('NER index: \n', list(ner_idx.items())[:10])

# We create the parallel sequences of indexes
X_idx = to_index(X_train_cat, word_idx)
Y_idx = to_index(Y_train_cat, ner_idx)

#print('First sentences, word indices \n', X_idx[:3])
#print('First sentences, NER indices \n', Y_idx[:3])

# We pad the sentences

X = pad_sequences(X_idx)
Y = pad_sequences(Y_idx)

print(X[0])
print(Y[0])

# The number of NER classes and 0 (padding symbol)
Y_train = to_categorical(Y, num_classes=len(ner) + 2)
print(Y_train[0])

# We create an embedding matrix
# 0 is the padding symbol and index one is a unknown word

rdstate = np.random.RandomState(1234567)
embedding_matrix = rdstate.uniform(-0.05, 0.05, 
                                   (len(vocabulary_words) + 2, 
                                    EMBEDDING_DIM))

for word in vocabulary_words:
    if word in embeddings_dict:
        # If the words are in the embeddings, we fill them with a value
        embedding_matrix[word_idx[word]] = embeddings_dict[word]

#print('Shape of embedding matrix:', embedding_matrix.shape)
#print('Embedding of table \n', embedding_matrix[word_idx['table']])
#print('Embedding of the padding symbol, idx 0, random numbers \n', 
#      embedding_matrix[0])

# ############ The Reccurent Network Model (Tagger)

model = models.Sequential()

model.add(layers.Embedding(len(vocabulary_words) + 2,      
                           EMBEDDING_DIM,
                           mask_zero=True,
                           input_length=None))

model.layers[0].set_weights([embedding_matrix])
# The default is True
model.layers[0].trainable = False

# a simple RNN network
#model.add(SimpleRNN(100, return_sequences=True))

# a simple RNN network with Bidirectional
#model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))

#a simple LSTM network
#model.add(LSTM(100, return_sequences=True))                         # dropout=0.1, recurrent_dropout=0.5,

# a stack of several recurrent layers
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Bidirectional(LSTM(100,
                             dropout=0.25,
                             recurrent_dropout=0.25,
                             return_sequences=True)))
model.add(Bidirectional(LSTM(100, return_sequences=True)))           # the last layer only returns the last output           

model.add(layers.Dropout(0.25))
model.add(Dense(NB_CLASSES + 2, activation='softmax'))


# Fitting the Model

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

model.fit(X, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# ####### Evaluation of system:

#Formatting the Test set
# In X_dict, we replace the words with their index
X_test_cat, Y_test_cat = build_sequences(test_dict)

# We create the parallel sequences of indexes
X_test_idx = to_index(X_test_cat, word_idx)
Y_test_idx = to_index(Y_test_cat, ner_idx)

print('X[0] test idx', X_test_idx[0])
print('Y[0] test idx', Y_test_idx[0])

X_test_padded = pad_sequences(X_test_idx)
Y_test_padded = pad_sequences(Y_test_idx)

print('X[0] test idx passed', X_test_padded[0])
print('Y[0] test idx padded', Y_test_padded[0])

# One extra symbol for 0 (padding)
Y_test_padded_vectorized = to_categorical(Y_test_padded, 
                                          num_classes=len(ner) + 2)
print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
print(X_test_padded.shape)
print(Y_test_padded_vectorized.shape)

# Evaluates with the padding symbol
test_loss, test_acc = model.evaluate(X_test_padded, 
                                     Y_test_padded_vectorized)
print('Loss:', test_loss)
print('Accuracy:', test_acc)

# We evaluate on all the test corpus

print('X_test' + '\n', X_test_cat[0])
print('X_test_padded' + '\n', X_test_padded[0])

corpus_ner_predictions = model.predict(X_test_padded)

print('Y_test' + '\n', Y_test_cat[0])
print('Y_test_padded' + '\n', Y_test_padded[0])
print('predictions' + '\n', corpus_ner_predictions[0])

# Remove padding
ner_pred_num = []

for sent_nbr, sent_ner_predictions in enumerate(corpus_ner_predictions):
    ner_pred_num += [sent_ner_predictions[-len(X_test_cat[sent_nbr]):]]
    
print(ner_pred_num[:2])

# Convert NER indices to symbols

ner_pred = []

for sentence in ner_pred_num:
    ner_pred_idx = list(map(np.argmax, sentence))
    #ner_pred_cat = list(map(rev_ner_idx.get, ner_pred_idx))
    ner_pred_cat = list(map(ner_rev_idx.get, ner_pred_idx))
    ner_pred += [ner_pred_cat]

print(ner_pred[:2])
print(len(ner_pred))
print(Y_test_cat[:2])

# Writting the results of our predictions and test set in one file
"""
After using the predict() method to predict the tags of the whole test set, we need to write our results in a file, where the two last columns will be the hand-annotated tag and the predicted tag. The fields must be separated by a space.
"""

def save(file, test_dict, column_names):
    """
    Saves the corpus in a file
    :param file:
    :param corpus_dict:
    :param column_names:
    :return:
    """
    with open(file, 'w') as f_out:
        for sentence in test_dict:
            sentence_lst = []
            
            for row in sentence:
                # The lambda technique is useful for example when we want to pass 
                # a simple function as an argument to another function, like this:
                items = map(lambda x: row.get(x, '_'), column_names)
                #sentence_lst += '\t'.join(items) + '\n'
                sentence_lst += ' '.join(items) + '\n'
                
            sentence_lst += '\n'
            f_out.write(''.join(sentence_lst))

testfile = 'NER-data/eng.test'

#column_names = ['id', 'form', 'lemma', 'cpos', 'pos', 'feats']
column_names_pred = ['form', 'pos', 'chunk', 'ner', 'predicted-ner']

testset = open(testfile).read().strip()

#conll_dict = CoNLLDictorizer(column_names, col_sep='\t')    # XXXXXX
conll_dict_pred = CoNLLDictorizer(column_names_pred, col_sep=' +')
test_dict_pred = conll_dict_pred.transform(testset)
print(len(test_dict_pred))

print(test_dict_pred[2])
#print(list(test_dict[:2]))
# word_idx = {v: k for k, v in rev_word_idx.items()} 

print((ner_pred[2]))
#pred_dict = {}
#sent_index = 0

print("hello",test_dict_pred[1])

for sent_index in range(len(test_dict_pred)):
#for sentence in test_dict_pred:
    #word_index = 0
    
    #if len(ner_pred[currentIndex]) != len(sentence):
    #    print("ERROR!")
    #ext_keys = []
    #ext_tags = []
    for word_index in range(len(test_dict_pred[sent_index])):
    #for word in sentence:   # every word is a row - dictionary
        #list(word.keys()).append('predicted_ner')
        #list(word.values()).append(str(ner_pred[sent_index][word_index]))
        #ext_keys = list(word.keys()).append('predicted_ner')
        #ext_tags = list(word.values()).append(str(ner_pred[sent_index][word_index]))
        #test_dict_pred = dict(zip(ext_keys, ext_tags))
        
        #pred_dict['predicted-ner'] = ner_pred[sent_index][word_index]
        #zip_dict = dict(zip(word, pred_dict))
        #word['predicted-ner'] = ner_pred[sent_index][word_index]
        test_dict_pred[sent_index][word_index]['predicted-ner'] = str(ner_pred[sent_index][word_index])
        #test_dict_pred.update({'predicted-ner': ner_pred[sent_index][word_index]})
        #sim_dict.update(word=sim)
        #value = ner_pred[sent_index][word_index]
        #word.update('predicted-ner'=value)
        #word_index += 1
    
    
    #sent_index += 1
    
    #if sent_index > 2000:
    #    break
    # i vårt test_dict, har vi meningar    
# para ihop dessa meningar med ner_pred mening
# para ihop orden i test-dict-meningen med tag i ner_pred mening   

# skriv ut file med: "ord", "GS", "pred-tag"

#print(type(zip_dict[:1]))
save('out', test_dict_pred, column_names_pred)
#save('out', zip_dict, column_names_pred)

# Evaluate
total, correct, total_ukn, correct_ukn = 0, 0, 0, 0

for id_s, sentence in enumerate(X_test_cat):
    for id_w, word in enumerate(sentence):
        total += 1
        if ner_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
            correct += 1
        # The word is not in the dictionary
        if word not in word_idx:
            total_ukn += 1
            if ner_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
                correct_ukn += 1

print('total %d, correct %d, accuracy %f' % 
      (total, correct, correct / total))
if total_ukn != 0:
    print('total unknown %d, correct %d, accuracy %f' % 
          (total_ukn, correct_ukn, correct_ukn / total_ukn))
