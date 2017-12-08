from subprocess import call

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import regularizers
import keras
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import data.twitter_user as twuser
import os, re
import pickle
import pandas as pd
import time
import numpy as np

dirname = os.path.dirname(__file__)

timesteps = 100
thought_vector_dimension = 2400
num_classes = 5
batch_size = 192
epochs = 100
train_dataset_size = 40000
val_dataset_size = 6000

'''
Encode tweets as thought vectors and then find its ten closest neighbors.
'''

TWEETS_TEST_DATA = "data/na/user_info.test"
TWEETS_DEV_DATA = "data/na/user_info.dev"
TWEETS_TRAIN_DATA = "data/na/user_info.train"

with open(os.path.join(dirname, "data/user_tweets_dev2.pickle"), 'rb') as handle:
    test_data = pickle.load(handle)

with open(os.path.join(dirname, "data/user_tweets_train2.pickle"), 'rb') as handle:
    train_data = pickle.load(handle)

train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
train_df = train_df.head(200000)
test_df = pd.DataFrame(test_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])

tokenizer = Tokenizer(num_words=50000, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
# tokenizer.fit_on_texts(np.append(train_df['tweets'].values, test_df['tweets'].values, axis=0))
tokenizer.fit_on_texts(train_df['tweets'].values)


X_train = tokenizer.texts_to_sequences(train_df['tweets'].values)
X_train = pad_sequences(X_train, maxlen=500, truncating='pre')
Y_train = train_df['region'].values
Y_train = keras.utils.to_categorical(Y_train, num_classes=5)

# tokenizer = Tokenizer(num_words=20000, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
# tokenizer.fit_on_texts(test_df['tweets'].values)
# X_test = tokenizer.texts_to_sequences(test_df['tweets'].values)
# X_test = pad_sequences(X_test, maxlen=400, truncating='pre')
# Y_test = test_df['region'].values
# Y_test = keras.utils.to_categorical(Y_test, num_classes=5)

model = Sequential()
model.add(Embedding(20000, 150))
model.add(LSTM(250, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(250, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

now = time.time()
log_dir = '/tmp/keras/{0}'.format(str(int(now)))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                   write_graph=True, batch_size=batch_size, write_images=True)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

lol = model.fit(X_train, Y_train, epochs=7, batch_size=batch_size,validation_split=.05,
                callbacks=[tensorboard_callback])
# model.fit_generator(
#     twuser.get_all_data_generator(twitter_users[0:train_dataset_size], batch_size=batch_size, timesteps=timesteps),
#     steps_per_epoch=train_dataset_size / batch_size,
#     validation_data=twuser.get_all_data_generator(
#         twitter_users[train_dataset_size:train_dataset_size + val_dataset_size], batch_size=batch_size,
#         timesteps=timesteps, dataset_type='val'),
#     validation_steps=val_dataset_size / batch_size,
#     epochs=epochs)

# lol = model.predict(data, verbose=True)
print("Hello")
