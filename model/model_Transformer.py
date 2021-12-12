import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from preprocessing import preprocessing
import sys
import os
import argparse
from nltk import CFG
from nltk.parse import RecursiveDescentParser, SteppingRecursiveDescentParser
from nltk.grammar import Production
import pandas as pd
import csv

script_dir = os.path.dirname(__file__)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

train = False

def main(argv):
    parser = argparse.ArgumentParser(prog="main", usage="%(prog)s --load_model <path/to/model>")
    parser.add_argument(
        "--evaluate", default=False, action="store_true", help="Training vs evaluate mode"
    )
    args = parser.parse_args()
    train = args.evaluate
    data = preprocessing()
    data = pd.read_csv('../data/nl_to_rlang_data.csv')
    X, y, option = data.natural_language, data.rlang, data.string_type
    text_pairs = []
    type_t = []
    for x, y, z in zip(X, y, option):
        text_pairs.append((x,y))
        type_t.append((x,y,z))

    random.shuffle(text_pairs)
    random.shuffle(type_t)
    num_val_samples = int(0.1 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]
    test_pairs_type = type_t[num_train_samples + num_val_samples :]

    item_length = len(test_pairs_type[0])

    with open('test_data.csv', 'w') as test_file:
        file_writer = csv.writer(test_file, delimiter='|')
        for i in range(item_length):
            file_writer.writerow([x[i] for x in test_pairs_type])

    vocab_size = 72
    sequence_length = 40
    batch_size = 64


    def custom_standardization(input_string):
        lowercase = tf.strings.lower(input_string)
        return lowercase


    nl_vectorization = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
    )
    rl_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
    )

    train_nl = [pair[0] for pair in train_pairs]
    train_rl = [pair[1] for pair in train_pairs]
    nl_vectorization.adapt(train_nl)
    rl_vectorization.adapt(train_rl)

    def format_dataset(nl, rl):
        nl = nl_vectorization(nl)
        rl = rl_vectorization(rl)
        return ({"encoder_inputs": nl, "decoder_inputs": rl[:, :-1],}, rl[:, 1:])


    def make_dataset(pairs):
        nl_texts, rl_texts = zip(*pairs)
        nl_texts = list(nl_texts)
        rl_texts = list(rl_texts)
        dataset = tf.data.Dataset.from_tensor_slices((nl_texts, rl_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset)
        return dataset.shuffle(2048).prefetch(16).cache()


    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    embed_dim = 256
    latent_dim = 2048
    num_heads = 8

    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    if train:
        epochs = 2  # This should be at least 30 for convergence

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="transformer_model_checkpoint.h5", 
            verbose=1, 
            save_weights_only=True,
            save_freq= 2243)

        transformer.summary()
        transformer.compile(
            "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"], 
        )
        transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cp_callback])

        transformer.save("transformer_model.h5")
    else:
        transformer.load_weights("transformer_model.h5")

        with open("test_data.csv") as file_name:
            file_read = csv.reader(file_name, delimiter="|")

            data = list(file_read) # TODO: add the part that pulls up the test data from the file

        evaluation_data = []
        for x, y, z in zip(data[0], data[1], data[2]):
            evaluation_data.append((x,y,z))

        print(evaluation_data)
        assert(evaluation_data != None)
        # Evaluation_data should be a tuple from nl to rl
        
        # evaluation_nl, evaluation_rl, statement_type = list(zip(*evaluation_data))
        rl_types = ["option", "policy"]
        parsers = {}
        for rl_type in rl_types:
            cfg_file = f'rlang_{rl_type}.cfg'
            grammar = CFG.fromstring(open(os.path.join(script_dir, "../generate/cfgs/" + cfg_file), 'r').read())
            parser = RecursiveDescentParser(grammar)
            parsers[rl_type] = parser
        
        # parsed = []
        for datum in evaluation_data:
            nl, rl, statement_type = datum
            sentence = rl.split()
            try:
                list(parser.parse(sentence))
                # for t in parser.parse(sentence):
                    
            except RecursionError as re:
                print("Unable to parse sentence; recursion error for ", sentence) 
                break

    rl_vocab = rl_vectorization.get_vocabulary()
    rl_index_lookup = dict(zip(range(len(rl_vocab)), rl_vocab))
    max_decoded_sentence_length = 40


    def decode_sequence(input_sentence):
        tokenized_input_sentence = nl_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = rl_vectorization([decoded_sentence])[:, :-1]
            predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = rl_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
        return decoded_sentence


    test_nl_texts = [pair[0] for pair in test_pairs]   
    test_rl_texts = [pair[1] for pair in test_pairs]
    for _ in range(30):
        rand = random.randint(0, len(test_nl_texts) -1)
        input_sentence = test_nl_texts[rand]
        expected_sentence = test_rl_texts[rand]
        translated = decode_sequence(input_sentence)

        print('Input Source sentence:', input_sentence)
        print('Actual Target Translation:', expected_sentence)
        print('Predicted Target Translation:', translated)
        print('')



class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
            'attention': self.attention,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'supports_masking': self.supports_masking,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'token_embeddings': self.token_embeddings,
            'position_embeddings': self.position_embeddings,
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'attention_1': self.attention_1,
            'attention_2': self.attention_2,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'layernorm_3': self.layernorm_3,
            'supports_masking': self.supports_masking,
        })
        return config

if __name__ == '__main__':
    main(sys.argv)
