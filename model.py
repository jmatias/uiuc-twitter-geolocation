from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers
import keras
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import data.twitter_user as twuser
import os, re

def purge(dir, pattern):
    if os.path.exists(dir):
        for f in os.listdir(dir):
            if re.search(pattern, f):
                os.remove(os.path.join(dir, f))


purge("/home/javier/harddrive/encodings/","encodings-")

timesteps = 100
thought_vector_dimension = 2400
num_classes = 5
batch_size = 200
epochs = 100
train_dataset_size = 40000
val_dataset_size = 6000

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

twitter_users = twuser.load_twitter_users(encoder, dataset='train')[0:train_dataset_size + val_dataset_size]

model = Sequential()
model.add(LSTM(1500, dropout=0.5, recurrent_dropout=0.5, input_shape=(timesteps, 2400)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(
    twuser.get_all_data_generator(twitter_users[0:train_dataset_size], batch_size=batch_size, timesteps=timesteps),
    steps_per_epoch=train_dataset_size / batch_size,
    validation_data=twuser.get_all_data_generator(
        twitter_users[train_dataset_size:train_dataset_size + val_dataset_size], batch_size=batch_size,
        timesteps=timesteps, dataset_type='val'),
    validation_steps=val_dataset_size / batch_size,
    epochs=epochs)

# lol = model.predict(data, verbose=True)
print("Hello")
