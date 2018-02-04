# -*- coding: utf-8 -*-
"""
Written by ZainNasrullah
"""

import pandas as pd
import numpy as np
import re
from os.path import isfile
from bs4 import BeautifulSoup
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

train = True
load_data = True
word_mode = False

# function to clean data
def clean_df(text):
    ''' clean the data frame to prepare for processing'''
    
    # remove any messages containing html tags as those do not contribute to speech
    nonsense_regex = re.compile(r'(<.*>)')
    
    # convert all html tags to unicode
    html_regex = re.compile(r'(&)')
    
    try:
        if nonsense_regex.search(text):
            clean_text = np.NaN
        elif html_regex.search(text):
            clean_text = BeautifulSoup(text, "html5lib").get_text()
        else:
            clean_text = text
    except TypeError:
        clean_text = np.NaN
        
    return clean_text

def generate_text(model, dataX, seq_length=10, word_mode=True):
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    if word_mode:
        print (' '.join([int_to_char[value] for value in pattern]))
    else:
        print (''.join([int_to_char[value] for value in pattern]))
    
    # generate characters
    for i in range(seq_length):
    	x = np.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model.predict(x, verbose=0)
    	index = np.argmax(prediction)
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]
    
    sequence = [int_to_char[value] for value in pattern]
    return sequence


source = 'skumbag-logs'
clean_source = source + '_clean.txt'

if isfile(clean_source):
    with open(clean_source, 'r') as file:
        raw_text = file.read()
else:
    # Read data set
    df = pd.read_csv(source+'.csv', header=None)
    df_text = df.iloc[:,6]
    
    # Clean dataset
    df_clean = df_text.apply(clean_df)
    df_clean = df_clean.dropna()
    
    text = df_clean.tolist()
    raw_text = ' '.join(text)
    raw_text = raw_text.replace('"', '')
    raw_text = ''.join([x for x in raw_text if ord(x) < 128])
    
    with open(clean_source, 'w') as file:
        file.write(raw_text)

# create mapping of unique chars to integers, and a reverse mapping
if word_mode:
    raw_text = raw_text.split()
    print("Word Mode")
else:  
   
    print("Character Mode")

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

print ("Total Words / Characters (Depending on Mode): ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
if isfile(source+'_X') and isfile(source+'_Y'):
    dataX = pickle.load(open(source+'_X', 'rb'))
    dataY = pickle.load(open(source+'_Y', 'rb'))
else:
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
    	seq_in = raw_text[i:i + seq_length]
    	seq_out = raw_text[i + seq_length]
    	dataX.append([char_to_int[char] for char in seq_in])
    	dataY.append(char_to_int[seq_out])
    
    pickle.dump(dataX, open(source+'_X', 'wb'))
    pickle.dump(dataY, open(source+'_Y', 'wb'))

n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add((Dense(y.shape[1], activation = 'softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath=source+"_weight"
if ( isfile(filepath) ):
    print("Loading Weights...")
    model.load_weights(filepath)
else:
    print("No weights found... Starting from scratch.")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

# fit the model
if train:
    model.fit(X, y, epochs=200, batch_size=16, callbacks=callbacks_list)
