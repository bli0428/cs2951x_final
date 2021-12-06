import pandas as pd
import numpy as np
import string
from string import digits
# import matplotlib.pyplot as plt
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from preprocessing import preprocessing
import sys


def prep_data(data, source_word2idx, target_word2idx, source_words, target_words):

    source_length_list=[]
    for l in data.natural_language:
        source_length_list.append(len(l.split(' ')))
    max_source_length= max(source_length_list)
    target_length_list=[]
    for l in data.rlang:
        target_length_list.append(len(l.split(' ')))
    max_target_length= max(target_length_list)

    X, y = data.natural_language, data.rlang
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    # Input tokens for encoder
    num_encoder_tokens=len(source_words) +1
    # Input tokens for decoder zero padded
    num_decoder_tokens=len(target_words) +1

    def generate_batch(X = X_train, y = y_train, batch_size = 128):
        ''' Generate a batch of data '''
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, max_source_length),dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_target_length),dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_target_length, num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        encoder_input_data[i, t] = source_word2idx[word] 
                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = target_word2idx[word] # decoder input seq
                        if t>0:
                            # decoder target sequence (one hot encoded)
                            # does not include the START_ token
                            # Offset by one timestep
                            #print(word)
                            decoder_target_data[i, t - 1, target_word2idx[word]] = 1.
                        
                yield([encoder_input_data, decoder_input_data], decoder_target_data)


    batch_size = 128
    epochs = 50
    latent_dim=256

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
    dec_emb = dec_emb_layer(decoder_inputs)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    train_samples = len(X_train) # Total Training samples
    val_samples = len(X_test)    # Total validation or test samples
    batch_size = 128
    epochs = 25

    model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                        steps_per_epoch = train_samples//batch_size,
                        epochs=epochs,
                        validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                        validation_steps = val_samples//batch_size)
    model.save_weights('model_weights_100epochs.h5')

def main(argv):
    data = preprocessing()

    nl_word_to_idx, rl_word_to_idx, nl_idx_to_word, rl_idx_to_word, source_words, target_words = build_mappings(data)
    data = shuffle(data)

    prep_data(data, nl_word_to_idx, rl_word_to_idx, source_words, target_words)

def build_mappings(data):
    all_source_words=set()
    for source in data.natural_language:
        for word in source.split():
            if word not in all_source_words:
                all_source_words.add(word)

    all_target_words=set()
    for target in data.rlang:
        for word in target.split():
            if word not in all_target_words:
                all_target_words.add(word)
    
    #create set of all target words
    source_words = sorted(list(all_source_words))
    target_words = sorted(list(all_target_words))

    #create word to index mappings
    nl_word_to_idx = dict([(word, i+1) for i,word in enumerate(source_words)])
    rl_word_to_idx = dict([(word, i+1) for i, word in enumerate(target_words)])

    #create index to word mappings
    nl_idx_to_word = dict([(i, word) for word, i in  nl_word_to_idx.items()])
    rl_idx_to_word = dict([(i, word) for word, i in rl_word_to_idx.items()])

    return nl_word_to_idx, rl_word_to_idx, nl_idx_to_word, rl_idx_to_word, source_words, target_words

if __name__ == '__main__':
    main(sys.argv)
