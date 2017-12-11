from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import time
import pickle
import numpy as np
from os import path
from data import constants


def top_5_acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


class Model:
    def __init__(self, num_outputs, use_tensorboard=False, batch_size=32, time_steps=500,
                 vocab_size=50000):
        """

        :param num_outputs: Number of output classes. For example, in the case of Census regions num of classes is 4.
        :param epochs:
        :param use_tensorboard: Track training progress using Tensorboard. Default: true.
        :param batch_size: Default: 32
        :param time_steps: Default: 500
        :param vocab_size: Use the top N most frequent words. Default: 50,000
        """
        self._epochs = None
        self._use_tensorboard = use_tensorboard
        self._batch_size = batch_size
        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._tokenizer = None
        self._tokenizer_cachefile = path.join(constants.DATACACHE_DIR, "tokenizer_cache.pickle")

        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, 150))
        self._model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        self._model.add(LSTM(300, dropout=0.5, recurrent_dropout=0.5))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])

    def _generate_callbacks(self):
        now = time.time()
        log_dir = './log_dir/{0}'.format(str(int(now)))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                           write_graph=True, batch_size=self._batch_size,
                                                           write_images=True)
        return [tensorboard_callback]

    def _create_tokenizer(self, x_train):

        if path.exists(self._tokenizer_cachefile):
            print("Loading cached tokenizer...")
            with open(self._tokenizer_cachefile, 'rb') as handle:
                self._tokenizer = pickle.load(handle)
        else:
            print("Building tweet Tokenizer using a {0} word vocabulary...".format(self._vocab_size))
            self._tokenizer = Tokenizer(num_words=self._vocab_size, lower=True,
                                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
            self._tokenizer.fit_on_texts(x_train)
            with open(self._tokenizer_cachefile, 'wb') as handle:
                pickle.dump(self._tokenizer, handle)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            if path.exists(self._tokenizer_cachefile):
                print("Loading cached tokenizer...")
                with open(self._tokenizer_cachefile, 'rb') as handle:
                    self._tokenizer = pickle.load(handle)
            else:
                raise Exception("Tokenizer has not been trained.")

    def _tokenize_texts(self, texts):
        self._load_tokenizer()
        texts = self._tokenizer.texts_to_sequences(texts)
        texts = pad_sequences(texts, maxlen=self._time_steps, truncating='pre')
        return texts

    def train(self, x_train, y_train, x_dev, y_dev, epochs=7):
        self._epochs = epochs

        print("Tokenizing tweets...")
        x_dev = self._tokenize_texts(x_dev)
        x_train = self._tokenize_texts(x_train)

        y_train = keras.utils.to_categorical(y_train, num_classes=self._num_outputs)
        y_dev = keras.utils.to_categorical(y_dev, num_classes=self._num_outputs)

        print("Training model...")
        history = self._model.fit(x_train, y_train, epochs=self._epochs, batch_size=self._batch_size,
                                  validation_data=(x_dev, y_dev), callbacks=self._generate_callbacks())
        return history

    def predict(self, x):
        self._load_tokenizer()
        x = self._tokenize_texts(x)
        return np.argmax(self._model.predict(x, batch_size=self._batch_size), axis=1)

    def evaluate(self, x_test, y_test):
        self._load_tokenizer()
        x_test = self._tokenize_texts(x_test)
        y_test = keras.utils.to_categorical(y_test, num_classes=self._num_outputs)
        return self._model.evaluate(x_test, y_test, batch_size=self._batch_size)

    def load_saved_model(self, filename):
        if not path.exists(filename):
            raise ValueError("Filename does not exist.", filename)
        self._model = keras.models.load_model(filename, custom_objects={'top_5_acc': top_5_acc})

    def save_model(self, filename):
        self._model.save(filename)
