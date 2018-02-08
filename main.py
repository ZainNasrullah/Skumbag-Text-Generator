"""
Written by Zain Nasrullah
"""
# Library Imports (General)
import pandas as pd
import numpy as np
import re
from os.path import isfile
from bs4 import BeautifulSoup
import pickle
import random
import sys

# Library Imports (Keras)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import Adam, RMSprop


def clean_df(text):
    ''' clean the data by removing html tags and artifacts'''

    # remove any messages containing html tags as those do not contribute to normal speech patterns
    nonsense_regex = re.compile(r'(<.*>)')

    # convert all html tags to unicode
    html_regex = re.compile(r'(&)')

    # Try removing html tags or replace with NaN upon a type error
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


def generate_text(model, dataX, input_sequence=None, seq_length=100, word_mode=False):
    ''' function to generate text from the model after training'''

    # generate seed (starting pattern) and print it out to the screen
    if input_sequence:
        pattern = [char_to_int[value] for value in input_sequence]
        pattern = pattern[-seq_length:]
    else:
        start = np.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]

    print("Seed:")
    if word_mode:
        print(' '.join([int_to_char[value] for value in pattern]))
    else:
        print(''.join([int_to_char[value] for value in pattern]))

    # generate characters
    for i in range(seq_length):

        # preprocess
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)

        # Predict probabilities and find index corresponding to largest probability
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)

        # append the index to pattern and then drop the first word/character
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # return the final sequence
    sequence = ''.join([int_to_char[value] for value in pattern])
    return sequence

# Settings (eventually move these into arguements)
train = True
word_mode = False
source = 'shakespeare'
clean_source = source + '_clean.txt'
seq_length = 60  # Sequence length to be used in training

# if the text file of the input exists read it, otherwise create from csv
# (note: original data is in .CSV form)
if isfile(clean_source):
    with open(clean_source, 'r') as file:
        raw_text = file.read()
else:
    # Read data set
    df = pd.read_csv(source + '.csv', header=None)
    df_text = df.iloc[:, 6]

    # Clean dataset
    df_clean = df_text.apply(clean_df)
    df_clean = df_clean.dropna()

    # Convert to list, separate tokens by spaces, remove quotes and any non-ascii values
    text = df_clean.tolist()
    raw_text = ' '.join(text)
    raw_text = raw_text.replace('"', '')
    raw_text = ''.join([x for x in raw_text if ord(x) < 128])

    # write output to a file to be loaded next time
    with open(clean_source, 'w') as file:
        file.write(raw_text)

# After preparing string, the data needs to be split if working with words
if word_mode:
    raw_text = raw_text.split()
    print("Word Mode")
else:
    print("Character Mode")

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Words / Characters (Depending on Mode): ", n_chars)
print("Total Vocab: ", n_vocab)

# Load the dataset if it already exists
if isfile(source + '_X') and isfile(source + '_Y'):
    X = pickle.load(open(source + '_X', 'rb'))
    y = pickle.load(open(source + '_Y', 'rb'))
else:
    # prepare the dataset of input to output pairs encoded as integers
    seq_in = []
    seq_out = []
    for i in range(0, n_chars - seq_length, 3):
       # Map k sequence to k+1 object
        seq_in.append(raw_text[i:i + seq_length])
        seq_out.append(raw_text[i + seq_length])
    
    # Create dataset representating a sequential one-hot-encoded vector
    X = np.zeros((len(seq_in), seq_length, n_vocab), dtype=np.bool)
    y = np.zeros((len(seq_in), n_vocab), dtype=np.bool)
    for i, sentence in enumerate(seq_in):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
            y[i, char_to_int[seq_out[i]]] = 1

    # save dataset to avoid overhead in repeatedly creating it
    pickle.dump(X, open(source + '_X', 'wb'))
    pickle.dump(y, open(source + '_Y', 'wb'))

# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(
    X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add((Dense(y.shape[1], activation='softmax')))

# define the checkpoint and load weight if applicible
filepath = source + "_weight"
if (isfile(filepath)):
    print("Loading Weights...")
    model.load_weights(filepath)
else:
    print("No weights found... Starting from scratch.")

# Compile model
adam_optimizer = Adam()
rms_optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer)

# Basically divides by temperature and transforms the output back into a probability model
# Then returns the argmax of a multinomial experiment with new predictions
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, n_chars - seq_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = raw_text[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, seq_length, n_vocab))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = int_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# Create call back list to checkpoint and print progress at the end of each epoch
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [checkpoint, print_callback]

# fit the model
if train:
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
