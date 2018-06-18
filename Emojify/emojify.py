# _*_ coding:utf-8 _*_
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model,load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w not in word_to_index.keys():
                X_indices[i, j] = len(word_to_index)
            else:
                X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1  #word index begin with 1,plus 1 for padding 0
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        try:
            emb_matrix[index, :] = word_to_vec_map[word]
        except:
            print(word)
            raise ValueError('Error')
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(16, activation='softmax')(X)
    X = Activation('softmax')(X)
    model = Model(sentence_indices, X)
    return model

X_train, Y_train = read_csv('data/short_sentences_train.csv')
maxLen = len(max(X_train, key=lambda x : len(x.split())).split())
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.twitter.27B.50d.txt')

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=16)

model.fit(X_train_indices, Y_train_oh, epochs=350, batch_size=32, shuffle=True)

# Use this for PopBots data
#x_test = np.array(['You are taking charge of the situation', 'Imagine a close friend came to you with similar problem', 'It is just not what I want right now', 'Perhaps I dont like the food I am eating', 'Oh sorry to hear Could you give me more information'])

x_test = np.array(['not feeling happy', 'you are so pretty', 'let us play ball'])

X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(x_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    print(' prediction: ' + x_test[i] + label_to_emoji(num).strip())

model.save('checkpoints/short_sentences_weights.h5')
