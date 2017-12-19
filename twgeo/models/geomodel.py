import pickle
import time
from os import path, environ

import keras
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing


def top_5_acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:
    """
    Geolocation prediction model. Consists of 4 layers ( Embedding, LSTM, LSTM and Dense).
    """

    def __init__(self, use_tensorboard=True, batch_size=64):
        """

        :param use_tensorboard: Track training progress using Tensorboard. Default: true.
        :param batch_size: Default: 64
        """
        self._use_tensorboard = use_tensorboard
        self._batch_size = batch_size
        self._tokenizer = None
        self._label_encoder = None

    def build_model(self, num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
        Build the model.

        :param num_outputs: Number of output classes. For example, in the case of Census regions num of classes is 4.
        :param time_steps: Default: 500
        :param vocab_size: Use the top N most frequent words. Default: 20,000
        :param hidden_layer_size: Number of neurons in the hidden layers. Default: 128
        :return:
        """

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size

        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))

        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, 150))
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])

    def train(self, x_train, y_train, x_dev, y_dev, epochs=7, reset_model=False):
        """
        Fit the model to the training data.

        :param x_train: Training samples.
        :param y_train: Training labels. Must be a vector of integer values.
        :param x_dev: Validation samples.
        :param y_dev: Validation labels. Must be a vector of integer values.
        :param epochs: Number of times to train on the whole data set. Default: 7
        :param reset_model: If this is set to True, it will discard any previously trained model and start from scratch.
        :return:
        :raises: ValueError: If the number of training samples and the number of labels do not match.
        """

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.", x_train.shape[0],
                             y_train.shape[0])

        if x_dev.shape[0] != y_dev.shape[0]:
            raise ValueError("x_dev and y_dev must have the same number of samples.", x_dev.shape[0],
                             y_dev.shape[0])

        if self._label_encoder is None: self._create_label_encoder(y_train)
        y_train = self._label_encoder.transform(y_train)
        y_dev = self._label_encoder.transform(y_dev)

        if self._tokenizer is None: self._create_tokenizer(x_train)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(x_train.shape[0] + x_dev.shape[0]))
        x_dev = self._tokenize_texts(x_dev)
        x_train = self._tokenize_texts(x_train)

        y_train = keras.utils.to_categorical(y_train, num_classes=self._num_outputs)
        y_dev = keras.utils.to_categorical(y_dev, num_classes=self._num_outputs)

        if self._use_tensorboard:
            callbacks = self._generate_callbacks()
        else:
            callbacks = []
        print("Training model...")
        history = self._model.fit(x_train, y_train, epochs=epochs, batch_size=self._batch_size,
                                  validation_data=(x_dev, y_dev), callbacks=callbacks)
        return history

    def predict(self, x):
        """
        Predict the location of the given samples.

        :param x: A vector of tweets. Each row corresponds to a single user.
        :return: The prediction results.
        """
        x = self._tokenize_texts(x)
        predictions = self._model.predict(x, batch_size=self._batch_size)

        predictions = np.argmax(predictions, axis=1)
        predictions = self._label_encoder.inverse_transform(predictions)
        return predictions

    def evaluate(self, x_test, y_test):
        """
        Get the loss, accuracy and top 5 accuracy of the model.

        :param x_test: Evaluation samples.
        :param y_test: Evaluation labels.
        :return: A dictionary of metric, value pairs.
        """

        x_test = self._tokenize_texts(x_test)

        y_test = self._label_encoder.transform(y_test)
        y_test = keras.utils.to_categorical(y_test, num_classes=self._num_outputs)
        metrics = self._model.evaluate(x_test, y_test, batch_size=self._batch_size)
        d = {}
        for i in range(len(self._model.metrics_names)):
            d[self._model.metrics_names[i]] = metrics[i]
        return d

    def load_saved_model(self, filename):
        """
        Load a previously trained model from disk.

        :param filename: The H5 model.
        :return:
        """
        model_filename = filename + ".h5"
        tokenizer_filename = filename + ".tokenizer"
        label_encoder_filename = filename + ".labelencoder"
        metadata_filename = filename + ".meta"

        if path.exists(model_filename):
            print("Loading saved model...")
        else:
            raise Exception("Saved model {0} does not exist.".format(model_filename), model_filename)

        self._model = keras.models.load_model(model_filename, custom_objects={'top_5_acc': top_5_acc})

        with open(tokenizer_filename, 'rb') as handle:
            self._tokenizer = pickle.load(handle)

        with open(label_encoder_filename, 'rb') as handle:
            self._label_encoder = pickle.load(handle)

        with open(metadata_filename, 'rb') as handle:
            metadata = pickle.load(handle)
            self._vocab_size = metadata['vocab_size']
            self._hidden_layer_size = metadata['hidden_layer_size']
            self._num_outputs = metadata['num_outputs']
            self._time_steps = metadata['time_steps']

    def save_model(self, filename):
        """
        Save the current model and trained weights for later use.

        :param filename: Prefix for the model filenames.
        """

        model_filename = filename + ".h5"
        tokenizer_filename = filename + ".tokenizer"
        label_encoder_filename = filename + ".labelencoder"
        metadata_filename = filename + ".meta"

        self._model.save(model_filename)

        with open(tokenizer_filename, 'wb') as handle:
            pickle.dump(self._tokenizer, handle)

        with open(label_encoder_filename, 'wb') as handle:
            pickle.dump(self._label_encoder, handle)

        metadata = {'hidden_layer_size': self._hidden_layer_size,
                    'vocab_size': self._vocab_size,
                    'num_outputs': self._num_outputs,
                    'time_steps': self._time_steps}

        with open(metadata_filename, 'wb') as handle:
            pickle.dump(metadata, handle)

    def _generate_callbacks(self):
        now = time.time()
        log_dir = './.tensorboard_dir/{0}'.format(str(int(now)))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                           write_graph=True, batch_size=self._batch_size,
                                                           write_images=True)
        return [tensorboard_callback]

    def _create_tokenizer(self, x_train):
        print("Building tweet Tokenizer using a {0:,} word vocabulary. This may take a while...".format(
            self._vocab_size))
        self._tokenizer = Tokenizer(num_words=self._vocab_size, lower=True,
                                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
        self._tokenizer.fit_on_texts(x_train)

    def _create_label_encoder(self, y_train):
        self._label_encoder = preprocessing.LabelEncoder()
        self._label_encoder.fit(y_train)

    def _tokenize_texts(self, texts):
        texts = self._tokenizer.texts_to_sequences(texts)
        texts = pad_sequences(texts, maxlen=self._time_steps, truncating='pre')
        return texts
