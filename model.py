from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras

import os
import pickle
import pandas as pd
import time

dirname = os.path.dirname(__file__)

timesteps = 100
thought_vector_dimension = 2400
num_classes = 53
batch_size = 192
epochs = 100

'''
Encode tweets as thought vectors and then find its ten closest neighbors.
'''

TWEETS_TEST_DATA = "data/na/user_info.test"
TWEETS_DEV_DATA = "data/na/user_info.dev"
TWEETS_TRAIN_DATA = "data/na/user_info.train"

with open(os.path.join(dirname, "data/user_tweets_train3.pickle"), 'rb') as handle:
    train_data = pickle.load(handle)

train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
train_df = train_df.head(400000)

tokenizer = Tokenizer(num_words=50000, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
tokenizer.fit_on_texts(train_df['tweets'].values)

X_train = tokenizer.texts_to_sequences(train_df['tweets'].values)
X_train = pad_sequences(X_train, maxlen=500, truncating='pre')
Y_train = train_df['state'].values
Y_train = keras.utils.to_categorical(Y_train, num_classes=53)

model = Sequential()
model.add(Embedding(50000, 150))
model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

now = time.time()
log_dir = '/tmp/keras/{0}'.format(str(int(now)))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                   write_graph=True, batch_size=batch_size, write_images=True)

lol = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=.025,
                callbacks=[tensorboard_callback])
print("Hello")
