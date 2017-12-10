from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import time
import pickle
import os


class Model:
    def __init__(self, epochs, train_datapath, use_tensorboard, batch_size, time_steps, vocab_size):
        self._epochs = epochs
        self._train_datpath = train_datapath
        self._use_tensorboard = use_tensorboard
        self._batch_size = batch_size
        self._time_steps = time_steps
        self._vocab_size = vocab_size

        self._num_classes = 53

        def top_5_acc(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, 150))
        self._model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        self._model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(self._num_classes, activation='softmax'))
        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy',top_5_acc])

    def _generate_callbacks(self):
        now = time.time()
        log_dir = './log_dir/{0}'.format(str(int(now)))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                           write_graph=True, batch_size=self._batch_size,
                                                           write_images=True)
        return [tensorboard_callback]

    def train(self, x_train, y_train):

        print("Building tweet Tokenizer using a {0} word vocabulary...".format(self._vocab_size))

        if (os.path.exists('data/tokenizer.pickle')):
            with open('data/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = Tokenizer(num_words=self._vocab_size, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
            tokenizer.fit_on_texts(x_train)
            with open('data/tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle)

        print("Tokenizing tweets...")

        if (os.path.exists('data/tokenized_train_tweets.pickle')):
            with open('data/tokenized_train_tweets.pickle', 'rb') as handle:
                x_train = pickle.load(handle)
        else:
            x_train = tokenizer.texts_to_sequences(x_train)
            x_train = pad_sequences(x_train, maxlen=self._time_steps, truncating='pre')
            with open('data/tokenized_train_tweets.pickle', 'wb') as handle:
                pickle.dump(x_train, handle)

        y_train = keras.utils.to_categorical(y_train, num_classes=self._num_classes)

        print("Training model...")
        history = self._model.fit(x_train, y_train, epochs=self._epochs, batch_size=self._batch_size,
                                  validation_split=0.025, callbacks=self._generate_callbacks())
        return history

    def predict(self):
        pass

    def load_saved_model(self):
        pass

    def save_model(self):
        self._model.save('geomodel_state.h5')
