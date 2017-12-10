from model import Model
import os
import pickle
import pandas as pd

geoModel = Model(epochs=5, batch_size=192, time_steps=500, train_datapath='data/user_tweets_train3.pickle',
                 use_tensorboard=True)

dirname = os.path.dirname(__file__)

with open(os.path.join(dirname, "data/user_tweets_train3.pickle"), 'rb') as handle:
    train_data = pickle.load(handle)

train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
train_df = train_df.head(50000)
X_train = train_df['tweets'].values
Y_train = train_df['state'].values

geoModel.train(X_train, Y_train)
