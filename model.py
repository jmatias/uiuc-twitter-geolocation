from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
import keras
import numpy as np
import pickle

with open("data/user_vector_means.dev", 'rb') as handle:
    tweets_dev = pickle.load(handle)

with open("data/user_vector_means.train", 'rb') as handle:
    tweets_train = pickle.load(handle)

data = list(tweets_train.values())
data = np.array(data)
input_x = data[:, 0:2400]
labels = data[:, 2400:2401]


data_val = list(tweets_dev.values())
data_ = np.array(data)
input_x_val = data[:, 0:2400]
labels_val = data[:, 2400:2401]

model = Sequential()
model.add(Dense(5000, activation='relu', input_dim=2400,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
one_hot_labels_val = keras.utils.to_categorical(labels_val, num_classes=5)

# Train the model, iterating on the data in batches of 32 samples
model.fit(input_x, one_hot_labels, epochs=10000, batch_size=512, validation_data=(input_x_val,one_hot_labels_val))

lol = model.predict(data, verbose=True)
print("Hello")
