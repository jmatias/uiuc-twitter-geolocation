from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers
import keras
import numpy as np
import pickle
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import data.twitter_user as twuser




timesteps = 200
thought_vector_dimension = 2400
num_classes = 55
batch_size = 128
epochs = 1000

'''
Encode tweets as thought vectors and then find its ten closest neighbors.
'''

TWEETS_TEST_DATA = "data/na/user_info.test"
TWEETS_DEV_DATA = "data/na/user_info.dev"
TWEETS_TRAIN_DATA = "data/na/user_info.train"

DATA_DIR = "/home/javier/harddrive/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/"
VOCAB_FILE = DATA_DIR + "vocab.txt"
EMBEDDING_MATRIX_FILE = DATA_DIR + "embeddings.npy"
CHECKPOINT_PATH = DATA_DIR + "model.ckpt-501424"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

twitter_users = twuser.load_twitter_users(encoder, dataset='dev')
train_x, train_y = twuser.get_all_data(twitter_users[0:3500])
# val_x, val_y = twuser.get_all_data(twitter_users[2000:2100])


# with open("data/user_vectors_x.train", 'rb') as handle:
#     train_x = pickle.load(handle)
#
# with open("data/user_vectors_y.train", 'rb') as handle:
#     train_y = pickle.load(handle)

val_x = train_x[3000:]
val_y = train_y[3000:]

train_x = train_x[0:3000]
train_y = train_y[0:3000]



# data = list(tweets_train.values())
# data = np.array(data)
# input_x = data[:, 0:thought_vector_dimension]
# labels = data[:, 2401:2402]
#
#
# data_val = list(tweets_dev.values())
# data_val_ = np.array(data)
# input_x_val = data_val_[:, 0:2400]
# labels_val = data_val_[:, 2401:2402]


# input_x = np.random.random((1000, timesteps, thought_vector_dimension))
# labels = np.random.random((1000, num_classes))

# Generate dummy validation data
# input_x_val = np.random.random((100, timesteps, thought_vector_dimension))
# labels_val = np.random.random((100, num_classes))

model = Sequential()
model.add(LSTM(500, dropout=0.5, recurrent_dropout=0.5, input_shape=(timesteps, 2400)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(train_y[:, 1], num_classes=num_classes)
one_hot_labels_val = keras.utils.to_categorical(val_y[:, 1], num_classes=num_classes)

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_x, one_hot_labels, epochs=epochs, batch_size=batch_size,
          validation_data=(val_x, one_hot_labels_val))

# lol = model.predict(data, verbose=True)
print("Hello")
